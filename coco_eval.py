from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import torch
import torch.backends.cudnn as cudnn

from core.single_stage_detector import SingleStageDetector
from datasets.coco_detection import make_still_coco
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



def main():
    args = parse_args()
    coco_path = '/home/etienneperot/workspace/data/coco/'
    _, dataset, classes = make_still_coco(coco_path, args.batchsize, args.num_workers)

    net = SingleStageDetector.mobilenet_v2_fpn(3, classes, act="sigmoid")
    if args.cuda:
        net.cuda()
        cudnn.benchmark = True

    results = []
    image_ids = []
    for data in tqdm(dataset, total=len(dataset)):
            inputs, targets, indices = data['data'], data['boxes'], data['idx']
            if args.cuda:
                inputs = inputs.cuda()

            net.reset()

            with torch.no_grad():
                preds = net.get_boxes(inputs, score_thresh=0.1)[0]

            for i in range(len(preds)):
                boxes, labels, scores = preds[i]
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    # append detection for each positively labeled class
                    image_result = {
                        'image_id': indices[i],
                        'category_id': dataset.label_to_coco_label(label),
                        'score': float(score),
                        'bbox': box.tolist(),
                    }
                    results.append(image_result)
                image_ids.append(indices[i])
    
    json.dump(results, open('{}_bbox_results.json'.format(dataset.set_name), 'w'), indent=4)

    # load results in COCO evaluation tool
    coco_true = dataset.coco
    coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(dataset.set_name))

    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()



if __name__ == '__main__':
    main()