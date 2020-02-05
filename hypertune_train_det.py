from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from datasets.moving_mnist import make_moving_mnist
from datasets.coco_detection import make_still_coco, make_moving_coco
from core.single_stage_detector import SingleStageDetector
from core.rpn import BoxHead

from core.trainer import DetTrainer
import optuna
from types import SimpleNamespace 


def get_optim(trial, net, cuda=True):
    #lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    #wd = trial.suggest_loguniform('weight_decay', 1e-6, 1e-4)
    #opt = trial.suggest_categorical('optim', ['Adam','SGD', 'AdamW'])
    lr, wd, opt = 5e-5, 3e-6, 'AdamW'

    optimizer = getattr(optim, opt)(net.parameters(), lr=lr, weight_decay=wd)
    if cuda:
        net.cuda()
        cudnn.benchmark = True

    start_epoch = 0  # start from epoch 0 or last epoch
  
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9, last_epoch=-1)

    out = SimpleNamespace()
    out.net = net
    out.optimizer = optimizer
    out.scheduler = scheduler
    out.start_epoch = start_epoch
    return out


from core.utils.opts import time_to_batch, batch_to_time, cuda_time
from core.modules import ConvLayer, PreActBlock, SequenceWise, ConvRNN, Bottleneck, BottleneckLSTM
from core.unet import UNet

# class TunaFPN(nn.Module):
#     def __init__(self, trial, cin=1):
#         super(TunaFPN, self).__init__()
#         self.cin = cin
#         self.base = 8

#         self.cout = trial.suggest_discrete_uniform('cout', 128, 512, 128)
#         self.levels = trial.suggest_discrete_uniform('levels', 3, 5, 1)

#         layer_type = trial.suggest_categorical('feedforward_layer', ['PreActBlock', 'ConvLayer', 'Bottleneck'])
#         kernel_size = trial.suggest_categorical('kernel_size', [3, 7])

#         layer_func = getattr(core.modules, layer_type)

#         if kernel_size == 7:
#             self.conv1 = SequenceWise(nn.Sequential(
#                 ConvLayer(cin, self.base * 2, kernel_size=7, padding=3, stride=2),
#                 layer_func(self.base * 2, self.base * 4, kernel_size=7, padding=3, stride=2),
#                 layer_func(self.base * 4, self.base * 8, kernel_size=7, padding=3, stride=2),
#             ))  
#         else:
#             self.conv1 = SequenceWise(nn.Sequential(
#                 ConvLayer(cin, self.base * 2, kernel_size=7, padding=3, stride=2),

#                 layer_func(self.base * 2, self.base * 4, kernel_size=3, stride=1),
#                 layer_func(self.base * 2, self.base * 4, kernel_size=3, stride=2),
                
#                 layer_func(self.base * 2, self.base * 4, kernel_size=3, stride=1),
#                 layer_func(self.base * 4, self.base * 8, kernel_size=3, stride=2)
#             ))  

#         down_layer_type = trial.suggest_categorical('donw_layer', ['PreActBlock', 'ConvLayer', 'ConvRNN'])
#         up_layer_type = trial.suggest_categorical('up_layer', ['PreActBlock', 'ConvLayer', 'ConvRNN'])
        
#         channel_list = [self.base * 8] * (self.levels-1) + [cout] * self.levels
#         down = lambda x, y: getattr(core.modules, down_layer_type)(x, y, stride=2)
#         up = lambda x, y: getattr(core.modules, up_layer_type)(x, y)
#         skip = lambda x, y: SequenceWise(nn.Conv2d(x, y, kernel_size=3, stride=1, padding=1))
#         self.conv2 = UNet(channel_list, trial.suggest_categorical('mode', ["sum","cat"]), down, up, skip, sequence_upsample)
#         #self.conv2 = UNet.recurrent_unet(channel_list, mode='cat')
        
#     def forward(self, x):
#         x1 = self.conv1(x)
#         outs = self.conv2(x1)[-self.levels:]
#         sources = [time_to_batch(item)[0] for item in outs][::-1]
#         return sources

#     def reset(self, mask=None):
#         for name, module in self._modules.items():
#             if hasattr(module, "reset"):
#                 module.reset(mask)
#             if hasattr(module, "reset_modules"):
#                 module.reset_modules(mask)



def define_model(trial):
    print('==> Building model..')
    #fg_threshold = trial.suggest_discrete_uniform('fg_threshold', 0.5, 0.7, 0.1)
    #bg_threshold = trial.suggest_discrete_uniform('bg_threshold', 0.3, 0.5, 0.1)
    #allow_low_quality_matches = trial.suggest_categorical('allow_low_quality_matches', [True, False])
    #loss = trial.suggest_categorical('act_loss', ['_focal_loss', '_ohem_loss'])
    
    fg_threshold = 0.5
    bg_threshold = 0.5
    allow_low_quality_matches = False
    loss = '_ohem_loss'
    
    act = 'sigmoid' if loss == '_focal_loss' else 'softmax'

    #backbone = define_backbone(trial) 
    # backbone = TunaFPN(trial, 3)
    # net = SingleStageDetector(backbone, BoxHead, 3, 10, act, ratios=[1.0], scales=[1.0, 1.5], loss=loss, fg_iou_threshold=fg_threshold,
    # bg_iou_threshold=bg_threshold, allow_low_quality_matches=allow_low_quality_matches)

    net = SingleStageDetector.mnist_vanilla_rnn(3, 10, act=act, loss=loss, fg_iou_threshold=fg_threshold,
    bg_iou_threshold=bg_threshold, allow_low_quality_matches=allow_low_quality_matches)
    out = get_optim(trial, net)
    return out


def objective(trial):
    out = define_model(trial)
    args = SimpleNamespace()
    args.cuda = True
    args.half = False
    args.clip_grad_norm = False
    args.log_every = 10

    budget = 100
    tbins = trial.suggest_discrete_uniform(1, 10)
    batchsize = budget // tbins

    trainer = DetTrainer('mnist/hypertune/', out.net, out.optimizer, out.scheduler)
    out.train, out.val, _ = make_moving_mnist(500, 10, tbins, 2, batchsize, 0, 128, 128)

    for epoch in range(out.start_epoch, 10):
        trainer.train(epoch, out.train, args)

    mean_ap = trainer.evaluate(out.start_epoch, out.val, args)
    return mean_ap


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))