# torch_object
ssd + convlstm

ssd part is mostly stolen from torch_cv: https://github.com/kuangliu/torchcv 
main difference is:

- rnn training
- adaptive anchor-boxes creation
- a simple simulator of moving boxes to train your rnn detector
