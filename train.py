from __future__ import division

import argparse, time, logging, random, math, os
import numpy as np
import mxnet as mx

from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.utils import makedirs, TrainingHistory
from gluoncv.data import transforms as gcv_transforms

from SlimCNN import SlimCNN, default_scales

if __name__ == "__main__":
    num_gpus = 1
    ctx = [mx.gpu(i) for i in range(num_gpus)]

    net = SlimCNN(3, 10, default_scales)
    net.initialize(ctx=ctx)

    transform_train = transforms.Compose([
        gcv_transforms.RandomCrop(32, pad=4),
        transforms.RandomFlipLeftRight(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    per_device_batch_size = 64
    num_workers = 4
    batch_size = per_device_batch_size * num_gpus

    train_data = gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=True).transform_first(transform_train),
        batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)

    val_data = gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=False).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    lr_decay = 0.1
    lr_decay_epoch = [20, 40, 50, np.inf]
    optimizer = 'nag'
    optimizer_params = {'learning_rate': 0.1, 'wd': 0.0001, 'momentum': 0.9}
    trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)

    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    train_metric = mx.metric.Accuracy()
    train_history = TrainingHistory(['training-error', 'validation-error'])

    def test(ctx, val_data):
        metric = mx.metric.Accuracy()
        for i, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            outputs = [net(X) for X in data]
            metric.update(label, outputs)
        return metric.get()
    
    epochs = 60
    lr_decay_count = 0

    for epoch in range(epochs):
        tic = time.time()
        train_metric.reset()
        train_loss = 0

        if epoch == lr_decay_epoch[lr_decay_count]:
            trainer.set_learning_rate(trainer.learning_rate*lr_decay)
            lr_decay_count += 1

        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)

            for k in range(len(default_scales)):
                for j in range(len(net.features)):
                    net.features[j].idx = k

                with ag.record():
                    output = [net(X) for X in data]
                    loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]

                for l in loss:
                    l.backward()

            trainer.step(batch_size, ignore_stale_grad=True)

            train_loss += sum([l.sum().asscalar() for l in loss])
            train_metric.update(label, output)

        name, acc = train_metric.get()
        name, val_acc = test(ctx, val_data)

        train_history.update([1-acc, 1-val_acc])
        print('[Epoch %d] train=%f val=%f loss=%f time: %f' % (epoch, acc, val_acc, train_loss, time.time()-tic))

    train_history.plot()

    for k, x in enumerate(default_scales):
        for j in range(len(net.features)):
            net.features[j].idx = k
        
        name, val_acc = test(ctx, val_data)
        print(x, val_acc)