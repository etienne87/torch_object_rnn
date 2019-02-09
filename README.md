# torch_object_rnn
ssd + convlstm

ssd part is mostly stolen from torch_cv: https://github.com/kuangliu/torchcv 

main differences:

- rnn training (requires a dataloader which streams temporally coherent batches)
- adaptive anchor-boxes creation
- a simple simulator of moving boxes to train your rnn detector (test this code right away)
- use of tensorboardX (install tf-nightly, tb-nightly, launch tensorboard --logdir runs)
