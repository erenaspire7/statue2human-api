from torch import nn
import functools

from model.uvcgan2.utils import get_activ_layer, get_norm_layer
from model.uvcgan2.layers.modnet import ModNet
from model.uvcgan2.layers.transformer import ExtendedPixelwiseViT


class PatchGANDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(
        self,
        image_channels=3,
        ndf=64,
        n_layers=3,
        norm="batch",
        max_mult=8,
        shrink_output=True,
        return_intermediate_activations=False,
    ):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(PatchGANDiscriminator, self).__init__()

        norm_layer = get_norm_layer(norm_type=norm)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1

        sequence = [
            nn.Conv2d(image_channels, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        nf_mult_prev = 1

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, max_mult)

            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, max_mult)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        self.model = nn.Sequential(*sequence)
        self.shrink_conv = None

        if shrink_output:
            self.shrink_conv = nn.Conv2d(
                ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw
            )

        self._intermediate = return_intermediate_activations

    def forward(self, input):
        """Standard forward."""
        z = self.model(input)

        if self.shrink_conv is None:
            return z

        y = self.shrink_conv(z)

        if self._intermediate:
            return (y, z)

        return y


class VitModNetGenerator(nn.Module):
    def __init__(
        self,
        features=384,
        n_heads=6,
        n_blocks=12,
        ffn_features=1536,
        embed_features=3384,
        activ="gelu",
        norm="layer",
        input_shape=(3, 256, 256),
        modnet_features_list=[48, 96, 192, 384],
        modnet_activ="leakyrelu",
        modnet_norm=None,
        modnet_downsample="conv",
        modnet_upsample="upsample-conv",
        modnet_rezero=False,
        modnet_demod=True,
        rezero=True,
        activ_output="sigmoid",
        style_rezero=True,
        style_bias=True,
        n_ext=1,
        **kwargs
    ):
        super().__init__(**kwargs)

        image_shape = input_shape

        self.image_shape = image_shape

        mod_features = features * n_ext

        self.net = ModNet(
            modnet_features_list,
            modnet_activ,
            modnet_norm,
            image_shape,
            modnet_downsample,
            modnet_upsample,
            mod_features,
            modnet_rezero,
            modnet_demod,
            style_rezero,
            style_bias,
            return_mod=False,
        )

        bottleneck = ExtendedPixelwiseViT(
            features,
            n_heads,
            n_blocks,
            ffn_features,
            embed_features,
            activ,
            norm,
            image_shape=self.net.get_inner_shape(),
            rezero=rezero,
            n_ext=n_ext,
        )

        self.net.set_bottleneck(bottleneck)

        self.output = get_activ_layer(activ_output)

    def forward(self, x):
        # x : (N, C, H, W)
        result = self.net(x)
        return self.output(result)
