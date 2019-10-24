from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import json
from pycocotools.cocoeval import COCOeval


def coco_eval(gts, proposals, dataset):
    """

    :param gts: list of K array of size 5 (xyxy label)
    :param proposals: list of K array of size 6 (xyxy score label)
    :return:
    """
    # start collecting results
    results = []
    image_ids = []

    for index, (gt, det) in enumerate(zip(gts, proposals)):
        data = dataset[index]
        scale = data['scale']
        boxes = det[:, :4]
        scores = det[:, 4]
        labels = det[:, 5]
        boxes /= scale

        if boxes.shape[0] > 0:
            # change to (x, y, w, h) (MS COCO standard)
            boxes[:, 2] -= boxes[:, 0]
            boxes[:, 3] -= boxes[:, 1]

        for box_id in range(boxes.shape[0]):
            score = float(scores[box_id])
            label = int(labels[box_id])
            box = boxes[box_id, :]

            # append detection for each positively labeled class
            image_result = {
                'image_id': dataset.image_ids[index],
                'category_id': dataset.label_to_coco_label(label),
                'score': float(score),
                'bbox': box.tolist(),
            }

            # append detection to results
            results.append(image_result)

        # append image to list of processed images
        image_ids.append(dataset.image_ids[index])

        # print progress
        print('{}/{}'.format(index, len(dataset)), end='\r')

    if not len(results):
        return

    # write output
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

    import pdb
    pdb.set_trace()

    print('test')