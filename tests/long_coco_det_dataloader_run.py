"""
Long Run of training loop to reproduce an error
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.optim as optim
from datasets.coco_detection import make_coco_dataset as make_still_coco
from core.single_stage_detector import SingleStageDetector
from tqdm import tqdm
from collections import deque

if __name__ == '__main__':
    coco_path = '/home/prophesee/work/etienne/datasets/coco/'
    batchsize = 32
    num_workers = 3
    train_dataset, test_dataset, classes = make_still_coco(coco_path, batchsize, num_workers)

    # net = SingleStageDetector(num_classes=classes, cin=3, act="sigmoid").cuda()
    # optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    history_loss = deque([], maxlen=50)
    for i in range(10):
        for j, data in tqdm(enumerate(train_dataset), total=len(train_dataset)):
            pass
            """ inputs, targets, reset = data
            inputs = inputs.cuda()
            net.reset()
            optimizer.zero_grad()
            loss_dict = net.compute_loss(inputs, targets)
            loss = sum([value for key, value in loss_dict.items()])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
            optimizer.step()
            history_loss.append(loss.item())
        print('last losses: ')
        print(history_loss) """



