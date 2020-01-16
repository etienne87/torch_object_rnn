from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch.optim as optim
import torch.backends.cudnn as cudnn

from datasets.moving_mnist import make_moving_mnist
from datasets.coco_detection import make_still_coco, make_moving_coco
from core.single_stage_detector import SingleStageDetector

from core.trainer import DetTrainer
import optuna
from types import SimpleNamespace 


def get_optim(trial, net, cuda=True):
    lr = trial.suggest_float('lr', -5, -2)
    wd = trial.suggest_int('weight_decay', -6, -3)

    opt = trial.suggest_categorical('optim', ['adam'], ['sgd'])
    
    optimizer = getattr(optim, opt)(net.parameters(), lr=lr, weight_decay=10**wd)
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


def define_model(trial):
    print('==> Building model..')
    fg_threshold = trial.suggest_float('fg_threshold', 0.5, 0.7)
    bg_threshold = trial.suggest_float('bg_threshold', 0.3, 0.5)
    allow_low_quality_matches = trial.suggest_int('allow_low_quality_matches', 0, 1)
    loss = trial.suggest_categorical('act_loss', ['focal_loss'], ['ohem_loss'])
    
    act = 'sigmoid' if loss == 'focal_loss' else 'softmax'
    net = SingleStageDetector.mnist_vanilla_rnn(3, 10, act=act, loss=loss, fg_iou_threshold=fg_threshold,
    bg_iou_threshold=bg_threshold, allow_low_quality_matches=allow_low_quality_matches)
    out = get_optim(trial, net)
    return out


def objective(trial):
    out = define_model(trial)
    args = SimpleNamespace()

    trainer = DetTrainer('mnist/hypertune/', out.net, out.optimizer, out.scheduler)
    out.train, out.val, _ = make_moving_mnist(500, 50, 8, 2, 8, 0, 256, 256)

    for epoch in range(out.start_epoch, 3):
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