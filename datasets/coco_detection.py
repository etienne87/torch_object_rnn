from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import torch
import numpy as np
import random

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data.sampler import Sampler
from datasets.coco_wrapper import COCO2 as COCO 

from core.utils import vis, opts, image as imutil, data_augment as da
import cv2



class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(self, root_dir, set_name='train2017', transform=None):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform
        self.coco = COCO(self.root_dir, self.set_name) 
        self.image_ids = self.coco.get_image_ids()
        self.load_classes()
        self.label_offset = 1

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.load_categories()
        categories.sort(key=lambda x: x['id'])
        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        self.labelmap = []
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)
            self.labelmap.append(c['name'])

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot, 'idx': idx}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        assert image_index < len(self.image_ids)
        image_info = self.coco.load_image(self.image_ids[image_index])
        path = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
        img = cv2.imread(path)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAYRGB)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0

    def load_annotations(self, image_index):
        # parse annotations
        coco_annotations = self.coco.load_annotation(self.image_ids[image_index])
        if len(coco_annotations) == 0:
            return coco_annotations

        annotations = np.zeros((0, 5))
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue
            
            # import pdb;pdb.set_trace()
            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id']) + self.label_offset
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.load_image(self.image_ids[image_index])
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return 80

    def build(self):
        pass

    def reset(self):
        pass


def collater(data):
    """collate images"""
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
    indices = [s['idx'] for s in data]

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = torch.from_numpy(img)
        annots[i] = torch.from_numpy(annots[i])

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    out = {'data':padded_imgs[None], 
           'boxes': [annots], 
           'resets': 1, 
           'indices': indices,
           'scales':  scales
           }
    return out


def video_collater(data):
    """collate videos of different sizes"""
    videos = [torch.from_numpy(s['img']).permute(0, 3, 1, 2) for s in data]
    batch_size = len(data)
    tbins = videos[0].shape[0]
    
    annots = [[torch.from_numpy(s['annot'][t]) for s in data] for t in range(tbins)]

    scales = [s['scale'] for s in data]
    indices = [s['idx'] for s in data]

    widths = [int(s.shape[-1]) for s in videos]
    heights = [int(s.shape[-2]) for s in videos]

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_videos = torch.zeros(tbins, batch_size, 3, max_height, max_width)

    for i in range(batch_size):
        height, width = heights[i], widths[i]
        padded_videos[:, i, :, :height, :width] = videos[i]
   
    out = {'data':padded_videos, 
           'boxes': annots, 
           'resets': 1, 
           'indices': indices,
           'scales':  scales
           }
    return out
    

class CameraMotion(object):
    """Simulate a camera motion in front of the image plane
       This Allows to make a fake video
    """
    def __init__(self, max_time=1):
        self.max_time = max_time
        self.border_mode = cv2.BORDER_CONSTANT

    def __call__(self, sample):
        image = sample['img']
        boxes = sample['annot']

        height, width = image.shape[:2]
        voyage = imutil.PlanarVoyage(height, width)

        if self.border_mode == cv2.BORDER_WRAP:
            boxes = imutil.wrap_boxes(boxes, height, width)

        imseq = []
        boxseq = []
        for _ in range(self.max_time):
            G_0to2 = voyage()
            out = cv2.warpPerspective(image, G_0to2, dsize=(width, height), borderMode=self.border_mode)
            labels = boxes[:, 4:5]
            tboxes = imutil.cv2_apply_transform_boxes(boxes[:, :4], G_0to2)
            tboxes = imutil.clamp_boxes(tboxes, height, width)
            tboxes = np.concatenate([tboxes, labels], 1)
            tboxes = imutil.discard_too_small(tboxes, 10)
            imseq.append(out[None])
            boxseq.append(tboxes)

        if self.max_time == 1:
            sample['img'] = imseq[0][0]
            sample['annot'] = boxseq[0]
        else:
            sample['img'] = np.concatenate(imseq)
            sample['annot'] = boxseq
        return sample


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, fixed_size=True, min_side=512, max_side=640):
        self.min_side = min_side
        self.max_side = max_side
        self.fixed_size = fixed_size
        self.interpolation = cv2.INTER_CUBIC

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = self.min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > self.max_side:
            scale = self.max_side / largest_side

        # resize the image with the computed scale
        height, width =  (int(round(rows * scale)), int(round((cols * scale))))
        image = cv2.resize(image, (width, height), interpolation=self.interpolation)
        rows, cols, cns = image.shape

        if self.fixed_size:
            new_image = np.zeros((self.max_side, self.max_side, cns)).astype(np.float32)
            new_image[:rows, :cols, :] = image.astype(np.float32)
        else:
            pad_w = 32 - rows % 32
            pad_h = 32 - cols % 32
            new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
            new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale

        return {'img': new_image, 'annot': annots, 
        'scale': scale, 'idx': sample['idx']}



class Flipper(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp
            sample['img'] = image
            sample['annot'] = annots
        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
        self.std = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        sample['img'] = ((image.astype(np.float32) - self.mean) / self.std)
        return sample


class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean is None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std is None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                range(0, len(order), self.batch_size)]

  

def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def viz_batch(data, unnormalize, labelmap, label_offset=1):
    for i in range(data['img'].shape[0]):
        img = np.array(255 * unnormalize(data['img'][i, :, :, :])).copy()
        img[img < 0] = 0
        img[img > 255] = 255
        img = np.transpose(img, (1, 2, 0))
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = data['annot'][i].cpu().numpy().astype(np.int32)
        bboxes = vis.boxarray_to_boxes(boxes, boxes[:, -1] - label_offset, labelmap)
        img_ann = vis.draw_bboxes(img, bboxes)
        cv2.imshow('im', img_ann)
        cv2.waitKey(0)


def viz_batch_video(data, unnormalize, labelmap, label_offset=1):
    data['data'] = data['data'].cpu()
    for i in range(data['data'].shape[1]):
        for t in range(data['data'].shape[0]):
            img = data['data'][t, i, :, :, :]
            img = np.array(255 * unnormalize(img)).copy()
            img[img < 0] = 0
            img[img > 255] = 255
            img = np.transpose(img, (1, 2, 0))
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes = data['boxes'][t][i].cpu().numpy().astype(np.int32)
            bboxes = vis.boxarray_to_boxes(boxes, boxes[:, -1] - label_offset, labelmap)
            img_ann = vis.draw_bboxes(img, bboxes)
            cv2.imshow('im', img_ann)
            cv2.waitKey(0)
    
def change_resolution(dataloader, size, fixed_size=False):
    dataloader.dataset.transform = transforms.Compose([
    Normalizer(), Flipper(), Resizer(min_side=size, max_side=size, fixed_size=fixed_size)])


def make_still_coco(root_dir, batchsize, num_workers, fixed_size=True):
    dataset_train = CocoDataset(root_dir, set_name='train2017', transform=transforms.Compose([
        da.DictWrapper(da.PhotometricDistort()), 
        Normalizer(), 
        Flipper(), 
        da.DictWrapper(da.CenterCrop()), 
        Resizer(fixed_size=fixed_size)]))
    dataset_val = CocoDataset(root_dir, set_name='val2017', 
                              transform=transforms.Compose([Normalizer(), Resizer(fixed_size=fixed_size)]))

    if fixed_size:
        train_loader = DataLoader(dataset_train, num_workers=num_workers, batch_size=batchsize,
                                collate_fn=collater, pin_memory=True)

        val_loader = DataLoader(dataset_val, num_workers=num_workers, batch_size=batchsize,
                                collate_fn=collater, pin_memory=True)
    else:
        train_sampler = AspectRatioBasedSampler(dataset_train, batch_size=batchsize, drop_last=False)
        train_loader = DataLoader(dataset_train, num_workers=num_workers, 
                                collate_fn=collater, batch_sampler=train_sampler, pin_memory=True)

        val_sampler = AspectRatioBasedSampler(dataset_val, batch_size=batchsize, drop_last=False)
        val_loader = DataLoader(dataset_val, num_workers=num_workers, 
                                collate_fn=collater, batch_sampler=val_sampler, pin_memory=True)
    # ensure preallocated cuda memory
    # train_loader = opts.WrapperSingleAllocation(train_loader, storage_size=batchsize*3*512*512)
    # val_loader = opts.WrapperSingleAllocation(val_loader, storage_size=batchsize*3*512*512)                   
    return train_loader, val_loader, len(dataset_train.labels)


def make_moving_coco(root_dir, batchsize, num_workers, fixed_size=True):
    dataset_train = CocoDataset(root_dir, set_name='train2017', transform=transforms.Compose([
        Normalizer(), Flipper(), Resizer(fixed_size=fixed_size), CameraMotion(10)]))
    dataset_val = CocoDataset(root_dir, set_name='val2017',
                              transform=transforms.Compose([Normalizer(), Resizer(fixed_size=fixed_size)]))

    train_sampler = AspectRatioBasedSampler(dataset_train, batch_size=batchsize, drop_last=False)
    train_loader = DataLoader(dataset_train, num_workers=num_workers,
                              collate_fn=video_collater, batch_sampler=train_sampler, pin_memory=True)

    val_sampler = AspectRatioBasedSampler(dataset_val, batch_size=batchsize, drop_last=False)
    val_loader = DataLoader(dataset_val, num_workers=num_workers,
                            collate_fn=video_collater, batch_sampler=val_sampler, pin_memory=True)
                
    return train_loader, val_loader, len(dataset_train.labels)


import time
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Coco Reader')
    parser.add_argument('path', type=str, help='where to save')
    parser.add_argument('--batchsize', type=int, default=8, help='batchsize')
    parser.add_argument('--video', action='store_true', help="viz videos or images")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    dataset_train, _, _ = make_still_coco(args.path, args.batchsize, 0)

    unnormalize = UnNormalizer()

    start = time.time()
    #t = tqdm(dataset_train, total=len(dataset_train))
    with tqdm(dataset_train, total=len(dataset_train)) as t:
        for batch_idx, data in enumerate(t):
            end = time.time()
            rt1 = end-start
            """ 
            if args.video:
                viz_batch_video(data, unnormalize, dataset_train.dataset.labels)
            else:
                data = {'img':data['data'][0].cpu(), 'annot':data['boxes'][0]}
                viz_batch(data, unnormalize, dataset_train.dataset.labelmap)
            """
            start = time.time()
            rt2 = start-end
            t.set_description('Loading Time %f Viz %f' % (rt1, rt2))
