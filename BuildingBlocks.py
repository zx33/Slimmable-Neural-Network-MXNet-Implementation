import time, os
import warnings
warnings.filterwarnings("ignore")
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
import mxnet.autograd as ag

class SlimmableNormalConv(nn.Block):
    def __init__(self, out_channel, in_channel, stride, scales, **kwargs):
        super(SlimmableNormalConv, self).__init__(**kwargs)
        self.stride = (stride, stride)
        self.idx = len(scales) - 1
        self.width_opt = []
        for x in scales:
            self.width_opt.append(int(out_channel * x))
        
        self.conv_weight = self.params.get('conv_weight', shape=(out_channel, in_channel, 3, 3), init=mx.init.MSRAPrelu())

        self.gamma = []
        self.beta = []
        self.moving_mean = []
        self.moving_var = []

        for i, x in enumerate(self.width_opt):
            self.gamma.append(self.params.get('gamma.{}'.format(i), shape=(x,), init=mx.init.Constant(1.0)))
            self.beta.append(self.params.get('beta.{}'.format(i), shape=(x,), init=mx.init.Constant(0.0)))
            self.moving_mean.append(self.params.get('moving_mean.{}'.format(i), shape=(x,), init=mx.init.Constant(0.0)))
            self.moving_var.append(self.params.get('moving_var.{}'.format(i), shape=(x,), init=mx.init.Constant(0.0)))

    def forward(self, x):
        inp = x.shape[1]
        oup = self.width_opt[self.idx]

        x = nd.Convolution(x, weight=self.conv_weight.data()[:oup,:inp,:,:], kernel=(3, 3), stride=self.stride, pad=(1, 1), num_filter=oup, no_bias=True)
        x = nd.BatchNorm(x, self.gamma[self.idx].data(), self.beta[self.idx].data(), self.moving_mean[self.idx].data(), self.moving_var[self.idx].data())
        x = nd.Activation(x, act_type='relu')

        return x

class GAPLinear(nn.Block):
    def __init__(self, out_unit, in_unit, **kwargs):
        super(GAPLinear, self).__init__(**kwargs)

        self.weight = self.params.get('weight', shape=(in_unit, out_unit), init=mx.init.MSRAPrelu())
        self.bias = self.params.get('bias', shape=(out_unit,), init=mx.init.Constant(0.0))
    
    def forward(self, x):
        inp = x.shape[1]
        x = nd.Pooling(x, global_pool=True)
        x = nd.flatten(x)
        x = nd.dot(x, self.weight.data()[:inp,:]) + self.bias.data()
        return x