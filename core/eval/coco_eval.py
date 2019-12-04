from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def summarize(coco_eval):
    '''
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    '''
    def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
        p = coco_eval.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap==1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = coco_eval.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,:,aind,mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = coco_eval.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,aind,mind]
        if len(s[s>-1])==0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s>-1])
        return mean_s
    def _summarizeDets():
        stats = np.zeros((12,))
        stats[0] = _summarize(1)
        stats[1] = _summarize(1, iouThr=.5, maxDets=coco_eval.params.maxDets[2])
        stats[2] = _summarize(1, iouThr=.75, maxDets=coco_eval.params.maxDets[2])
        stats[3] = _summarize(1, areaRng='small', maxDets=coco_eval.params.maxDets[2])
        stats[4] = _summarize(1, areaRng='medium', maxDets=coco_eval.params.maxDets[2])
        stats[5] = _summarize(1, areaRng='large', maxDets=coco_eval.params.maxDets[2])
        stats[6] = _summarize(0, maxDets=coco_eval.params.maxDets[0])
        stats[7] = _summarize(0, maxDets=coco_eval.params.maxDets[1])
        stats[8] = _summarize(0, maxDets=coco_eval.params.maxDets[2])
        stats[9] = _summarize(0, areaRng='small', maxDets=coco_eval.params.maxDets[2])
        stats[10] = _summarize(0, areaRng='medium', maxDets=coco_eval.params.maxDets[2])
        stats[11] = _summarize(0, areaRng='large', maxDets=coco_eval.params.maxDets[2])
        return stats
    def _summarizeKps():
        stats = np.zeros((10,))
        stats[0] = _summarize(1, maxDets=20)
        stats[1] = _summarize(1, maxDets=20, iouThr=.5)
        stats[2] = _summarize(1, maxDets=20, iouThr=.75)
        stats[3] = _summarize(1, maxDets=20, areaRng='medium')
        stats[4] = _summarize(1, maxDets=20, areaRng='large')
        stats[5] = _summarize(0, maxDets=20)
        stats[6] = _summarize(0, maxDets=20, iouThr=.5)
        stats[7] = _summarize(0, maxDets=20, iouThr=.75)
        stats[8] = _summarize(0, maxDets=20, areaRng='medium')
        stats[9] = _summarize(0, maxDets=20, areaRng='large')
        return stats
    if not coco_eval.eval:
        raise Exception('Please run accumulate() first')
    iouType = coco_eval.params.iouType
    if iouType == 'segm' or iouType == 'bbox':
        summarize = _summarizeDets
    elif iouType == 'keypoints':
        summarize = _summarizeKps
    stats = summarize()
    return stats
    


def coco_eval(gts, proposals, labelmap, height, width, tmp_path, epoch):
    categories = [{"id": id + 1, "name": class_name, "supercategory": "none"}
                                for id, class_name in enumerate(labelmap)]

    annotations = []
    results = []
    image_ids = []
    images = []
    box_type = np.float32
    for image_id, (gt, pred) in enumerate(zip(gts, proposals)):
        im_id = image_id + 1

        images.append(
            {"date_captured" : "2019",
            "file_name" : "n.a", 
            "id" : im_id,
            "license" : 1,
            "url" : "",
            "height" : height,
            "width" : width})

        
        for i in range(len(gt)):
            bbox = gt[i]
            segmentation = []
            x1, y1, x2, y2 = bbox[:4].astype(box_type).tolist()
            w, h = (x2-x1), (y2-y1)
            area = w * h
            category_id = bbox[4] 
            annotation = {
                "area" : float(area),
                "iscrowd" : False,
                "image_id" : im_id,
                "bbox" : [x1, y1, w, h],
                "category_id" : int(category_id) + 1,
                "id": len(annotations) + 1 
            }
            annotations.append(annotation)
        
        for i in range(len(pred)):
            bbox = pred[i, :4]

            x1, y1, x2, y2 = bbox[:4].astype(box_type).tolist()
            w, h = (x2-x1), (y2-y1)

            score = pred[i, 4]
            category_id = pred[i, 5]
            image_result = {
                            'image_id': im_id,
                            'category_id': int(category_id) + 1,
                            'score': float(score),
                            'bbox': [x1, y1, w, h],
                        }
            results.append(image_result)

        image_ids.append(im_id)

    json_data = {"info" : {},
                "licenses" : [],
                "type" : 'instances',
                "images" : images,
                "annotations" : annotations,
                "categories" : categories}

    gt_filename = os.path.join(tmp_path, 'gt.json')
    result_filename = os.path.join(tmp_path, 'res.json')
    json.dump(json_data, open(gt_filename, 'w'), sort_keys=True, indent=4)
    json.dump(results, open(result_filename, 'w'), indent=4)

    coco_true = COCO(gt_filename)
    coco_pred = coco_true.loadRes(result_filename)
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids 
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.stats = summarize(coco_eval)

    stats = {
        "mean_ap": coco_eval.stats[0],
        "mean_ap50": coco_eval.stats[1],
        "mean_ap75": coco_eval.stats[2],
        "mean_ap_small": coco_eval.stats[3],
        "mean_ap_medium":coco_eval.stats[4],
        "mean_ap_big": coco_eval.stats[5]
    }
    return stats


if __name__ == '__main__':
    from core.eval.test import generate_dummy_imgs, show_gt_pred

    height, width = 512, 512
    num_classes = 10
    num_images = 20
    max_box = 10
    labelmap = [str(i) for i in range(num_classes)]
    gts, proposals = generate_dummy_imgs(num_imgs=num_images, num_classes=num_classes, max_box=max_box)

    stats = coco_eval(gts, proposals, labelmap, height, width, tmp_path="")
    print(stats)
