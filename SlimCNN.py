import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
import mxnet.autograd as ag

from BuildingBlocks import SlimmableNormalConv, GAPLinear

default_scales = [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]

class SlimCNN(nn.Block):
    def __init__(self, n=3, classes=10, scales=default_scales, **kwargs):
        super(SlimCNN, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.Sequential(prefix='plaincnn')

            self.features.add(SlimmableNormalConv(32, 3, 1, scales))
            for _ in range(1, n):
                self.features.add(SlimmableNormalConv(32, 32, 1, scales))
            
            self.features.add(SlimmableNormalConv(64, 32, 2, scales))
            for _ in range(1, n):
                self.features.add(SlimmableNormalConv(64, 64, 1, scales))

            self.features.add(SlimmableNormalConv(128, 64, 2, scales))
            for _ in range(1, n):
                self.features.add(SlimmableNormalConv(128, 128, 1, scales))

            self.output = GAPLinear(classes, 128)
    
    def forward(self, x):
        x = self.features(x)
        x = self.output(x)

        return x