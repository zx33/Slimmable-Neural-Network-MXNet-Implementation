# Slimmable Neural Network MXNet Implementation

A gluon (mxnet) implementation of Slimmable Neural Network and its derivative.

## Description

This code gives a example of Slimmable PlainCNN which contains 9 convolutional layers.

```python
image -> 32 -> 32 -> 32 -> (64, 2) -> 64 -> 64 -> (128, 2) -> 128 -> 128 -> outputs
```

The network has been trained for 60 epochs on CIFAR-10, and gets the final accuracy of 89.74%.

Scals | Accuracy
:-:|:-:
0.25 | 26.06%
0.375 | 35.37%
0.5 | 45.84%
0.625 | 59.52%
0.75 | 75.34%
0.875 | 83.82%
1.0 | 89.74%

> **BuildingBlocks.py** contains the building blocks of Slimmable Neural Network, such as **SlimmableNormalBlock** and **GAPLinear**.
> 
> **SlimCNN.py** is the code of Slimmable PlainCNN.
> 
> **train.py** is the code of training which is modified from the Gluon-cv official training code for CIFAR-10.
> 
> **train_val_loss.png** is the error curve of the example training process.