from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob


def time_to_batch(x):
    t, n = x.size()[:2]
    x = x.view(n * t, *x.size()[2:])
    return x, n

def batch_to_time(x, n=32):
    nt = x.size(0)
    time = nt // n
    x = x.view(time, n, *x.size()[1:])
    return x

def load_last_checkpoint(logdir, net, optimizer=None):
    checkpoints = glob.glob(logdir + '/checkpoints/' + '*.pth')
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('checkpoint#')[1].split('.pth')[0]))
    last_checkpoint = checkpoints[-1]
    checkpoint = torch.load(last_checkpoint)
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return start_epoch


def video_collate_fn(data_list):
    videos, boxes = zip(*data_list)
    videos = torch.stack(videos, 1)
    t, n = videos.shape[:2]
    boxes = [[boxes[i][t] for i in range(n)] for t in range(t)]
    return {'data': videos, 'boxes': boxes}

def image_collate_fn(data_list):
    images, boxes = zip(*data_list)
    images = torch.stack(images, 0)
    return {'data': images, 'boxes': boxes}


def video_collate_fn_with_reset_info(data_list):
    videos, boxes, resets = zip(*data_list)
    videos = torch.stack(videos, 1)
    t, n = videos.shape[:2]
    boxes = [[boxes[i][t] for i in range(n)] for t in range(t)]
    return {'data': videos, 'boxes': boxes, 'resets': sum(resets)>0}

def cuda_tick():
    torch.cuda.synchronize()
    return time.time()

def cuda_time(func):
    def wrapper(*args, **kwargs):
        start = cuda_tick()
        out = func(*args, **kwargs)
        end = cuda_tick()
        print(end-start, ' s @', func)
        return out
    return wrapper

class WrapperSingleAllocation(object):
    """ Receives batch from dataloader
        Applies list of functions at the batch level
    """
    def __init__(self, dataloader, storage_size):
        self.storage = torch.Tensor(storage_size).cuda().fill_(0)
        self.dataloader = dataloader
        self.dataset = self.dataloader.dataset #short-cut

    def __len__(self):
        return len(self.dataloader)
    
    def __iter__(self):
        for data in self.dataloader:
            data_tensor_cpu = data['data']
            shape = data_tensor_cpu.size()
            stride = data_tensor_cpu.stride()
            y = torch.Tensor().cuda()
            y.set_(self.storage.storage(), storage_offset=0, size=shape, stride=stride)
            y[...] = data_tensor_cpu
            data['data'] = y
            yield data  