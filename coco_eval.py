from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import json

import torch
import torch.backends.cudnn as cudnn

from core.single_stage_detector import SingleStageDetector
from core.utils import opts
from datasets.coco_detection import make_coco_dataset as make_still_coco
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch SSD Training')
    parser.add_argument('logdir', type=str, help='where to save')
    parser.add_argument('--path', type=str, default='', help='path to dataset')
    parser.add_argument('--batchsize', type=int, default=8, help='batchsize')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    parser.add_argument('--num_workers', type=int, default=2, help='save_every')
    return parser.parse_args()


def xyxytoxywh(bbox):
    x1, y1, x2, y2 = bbox.tolist()
    w, h = (x2-x1), (y2-y1)
    return [x1, y1, w, h]

def main():
    args = parse_args()
    coco_path = '/home/etienneperot/workspace/data/coco/'
    coco_path = '/home/prophesee/work/etienne/datasets/coco/'
    
    _, dataloader, classes = make_still_coco(coco_path, args.batchsize, args.num_workers)


    net = SingleStageDetector.mobilenet_v2_fpn(3, classes, act="sigmoid")
    if args.cuda:
        net.cuda()
        cudnn.benchmark = True
    net.eval()


    print('==> Resuming from checkpoint..')
    start_epoch = opts.load_last_checkpoint(args.logdir, net) + 1
    print('Epoch: ', start_epoch)

    gts = []
    results = []
    image_ids = []
    for batch_idx, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs, targets, indices = data['data'], data['boxes'], data['indices']

        image_indices = [dataloader.dataset.image_ids[item] for item in indices]
        if args.cuda:
            inputs = inputs.cuda()

        net.reset()

        with torch.no_grad():
            preds = net.get_boxes(inputs, score_thresh=0.1)[0]

        for i in range(len(preds)):
            boxes, labels, scores = preds[i]
            num_boxes = 0 if boxes is None else boxes.shape[0] 
            for box_id in range(num_boxes):
                score = float(scores[box_id])
                label = int(labels[box_id])
                box = boxes[box_id, :]
                box_xywh = xyxytoxywh(box)

                # append detection for each positively labeled class
                image_result = {
                    'id': indices[i],
                    'image_id': image_indices[i],
                    'category_id': dataloader.dataset.label_to_coco_label(label),
                    'score': float(score),
                    'bbox': box_xywh,
                }
                results.append(image_result)
            image_ids.append(indices[i])
        
        for i in range(len(targets[0])):
            boxes = targets[0][i]
            num_boxes = boxes.shape[0]
            for box_id in range(num_boxes):
                label = int(boxes[box_id, 4])-dataloader.dataset.label_offset
                box = boxes[box_id, :4]
                box_xywh = xyxytoxywh(box)
                area = box_xywh[2] * box_xywh[3]

                # append detection for each positively labeled class
                image_gt = {
                    'id': indices[i],
                    'image_id': image_indices[i],
                    'category_id': dataloader.dataset.label_to_coco_label(label),
                    'bbox': box_xywh,
                    'iscrowd': False,
                    'area': area
                }
                gts.append(image_gt) 
        

    original_gt_path = os.path.join(coco_path, 'annotations', 'instances_val2017.json')
    with open(original_gt_path, 'r') as fp:
        original_dict = json.load(fp)
    original_dict['annotations'] = gts  

    result_path = '{}_bbox_results.json'.format(dataloader.dataset.set_name)
    gt_path = '{}_bbox_gts.json'.format(dataloader.dataset.set_name)
    ids_path = 'image_ids.json'
    json.dump(results, open(result_path, 'w'), indent=4)
    json.dump(original_dict, open(gt_path, 'w'), indent=4)
    json.dump(image_ids, open(ids_path, 'w'), indent=4) 
    
    with open(ids_path, 'r') as fp:
        image_ids = json.load(fp)

    # load results in COCO evaluation tool
    coco_true = COCO(gt_path)
    coco_pred = coco_true.loadRes(result_path)

    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids 
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()



if __name__ == '__main__':
    main()
