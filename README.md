# torch_object_rnn
ssd + convlstm

some code is taken from from torch_cv: https://github.com/kuangliu/torchcv

main differences:

- rnn training (the moving_mnist dataloader is an example which streams temporally coherent batches
- a simple simulator of moving digits to train the rnn detector
- use of tensorboardX (install tf-nightly, tb-nightly, launch tensorboard --logdir runs)


![](data/moving_mnist_detection.gif)

![](data/focal_softmax_vs_sigmoid.png)