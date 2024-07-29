from torch import nn
from model.uvcgan2.utils import get_norm_layer, get_activ_layer, get_downsample_x2_layer


class UnetBasicBlock(nn.Module):

    def __init__(
        self, in_features, out_features, activ, norm, mid_features=None, **kwargs
    ):
        super().__init__(**kwargs)

        if mid_features is None:
            mid_features = out_features

        self.block = nn.Sequential(
            get_norm_layer(norm, in_features),
            nn.Conv2d(in_features, mid_features, kernel_size=3, padding=1),
            get_activ_layer(activ),
            get_norm_layer(norm, mid_features),
            nn.Conv2d(mid_features, out_features, kernel_size=3, padding=1),
            get_activ_layer(activ),
        )

    def forward(self, x):
        return self.block(x)


class UNetEncBlock(nn.Module):

    def __init__(self, features, activ, norm, downsample, input_shape, **kwargs):
        super().__init__(**kwargs)

        self.downsample, output_features = get_downsample_x2_layer(downsample, features)

        (C, H, W) = input_shape
        self.block = UnetBasicBlock(C, features, activ, norm)

        self._output_shape = (output_features, H // 2, W // 2)

    @property
    def output_shape(self):
        return self._output_shape

    def forward(self, x):
        r = self.block(x)
        y = self.downsample(r)
        return (y, r)
