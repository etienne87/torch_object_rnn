import os
import time
import torch
import numpy as np
import cv2
from multiprocessing import Queue, Process

from tensorboardX.summary import _clean_tag
from tensorboardX.proto.summary_pb2 import Summary
import imageio
import tempfile

def boxarray_to_boxes(boxes, labels, labelmap):
    boxes = boxes.cpu().numpy().astype(np.int32)
    labels = labels.cpu().numpy().astype(np.int32)
    bboxes = []
    for label, box in zip(labels, boxes):
        class_name = labelmap[label]
        pt1 = (box[0], box[1])
        pt2 = (box[2], box[3])
        bb = (class_name, label, pt1, pt2, None, None, None)
        bboxes.append(bb)
    return bboxes


# Training
def encode_boxes(targets, box_coder, cuda):
    loc_targets, cls_targets = [], []

    for t in range(len(targets)):
        for i in range(len(targets[t])):
            boxes, labels = targets[t][i][:, :-1], targets[t][i][:, -1]
            if cuda:
                boxes, labels = boxes.cuda(), labels.cuda()
            loc_t, cls_t = box_coder.encode(boxes, labels)
            loc_targets.append(loc_t.unsqueeze(0))
            cls_targets.append(cls_t.unsqueeze(0).long())
            
    loc_targets = torch.cat(loc_targets, dim=0)  # (N,#anchors,4)
    cls_targets = torch.cat(cls_targets, dim=0)  # (N,#anchors,C)
    
    if cuda:
        loc_targets = loc_targets.cuda()
        cls_targets = cls_targets.cuda()
        
    return loc_targets, cls_targets


# batch to time for rank 3 tensors
def dig_out_time(x, n=32):
    nt, nanchors, c = x.size()
    t = int(nt / n)
    x = x.view(t, n, nanchors, c)
    return x


def single_frame_display(im):
    return make_single_channel_display(im[0], -1, 1)


def prepare_ckpt_dir(filename):
    dir = os.path.dirname(filename)
    if not os.path.isdir(dir):
        os.mkdir(dir)


def make_video(tensor, fps):
    t, h, w, c = tensor.shape
    with tempfile.NamedTemporaryFile() as f:
        filename = f.name + '.gif'
    images = [tensor[t] for t in range(tensor.shape[0])]
    imageio.mimwrite(filename, images, duration=0.04)
    with open(filename, 'rb') as f:
        tensor_string = f.read()
    try:
        os.remove(filename)
    except OSError:
        pass

    return Summary.Image(height=h, width=w, colorspace=c, encoded_image_string=tensor_string)


def add_video(writer, tag, tensor_thwc, global_step=None, fps=30, walltime=None):
    """found that add_video from tbX is buggy"""
    tag = _clean_tag(tag)
    video = make_video(tensor_thwc, fps)
    summary = Summary(value=[Summary.Value(tag=tag, image=video)])
    writer.file_writer.add_summary(summary, global_step, walltime)


class StreamDataset(object):
    """This class streams data in parallel to your main processing.
       Only 1 thread is used.
       Requires that your class has a "next"
    """
    def __init__(self, source, max_iter):
        self.source = source
        self.max_iter = max_iter

    def worker(self, q):
        while (1):
            q.put(self.source.next())
            time.sleep(0.01)

    def __iter__(self):
        q = Queue(maxsize=5)
        p = Process(name='daemon', target=self.worker, args=(q,))
        p.start()
        for i in range(self.max_iter):
            x, y = q.get()
            yield x, y
        p.terminate()

    def __len__(self):
        return self.max_iter


#UI Utils
def draw_bboxes(img, bboxes):
    colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)

    colors = [tuple(*item) for item in colors.tolist()]
    for bbox in bboxes:
        class_name, class_id, pt1, pt2, _, _, _ = bbox
        center = ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)
        color = colors[class_id * 60]
        cv2.rectangle(img, pt1, pt2, color, 2)
        cv2.putText(img, class_name, (center[0], max(0, pt2[1] - 1) ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
    return img


def filter_outliers(g, num_std=2):
    grange = num_std*g.std()
    gimg_min = g.mean() - grange
    gimg_max = g.mean() + grange
    g_normed = np.minimum(np.maximum(g,gimg_min),gimg_max) #clamp
    return g_normed


def make_single_channel_display(img, low=None, high=None):
    if low is None:
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    else:
        img = ((img+1)/2*255).astype(np.uint8)
    img = (img.astype(np.uint8))[..., None].repeat(3, 2)
    return img