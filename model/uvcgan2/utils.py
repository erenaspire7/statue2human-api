from torch import nn
import copy
import functools


class Identity(nn.Module):
    def forward(self, x):
        return x


def extract_name_kwargs(obj):
    if isinstance(obj, dict):
        obj = copy.copy(obj)
        name = obj.pop("name")
        kwargs = obj
    else:
        name = obj
        kwargs = {}

    return (name, kwargs)


def get_activ_layer(activ):
    name, kwargs = extract_name_kwargs(activ)

    if (name is None) or (name == "linear"):
        return nn.Identity()

    if name == "gelu":
        return nn.GELU(**kwargs)

    if name == "relu":
        return nn.ReLU(inplace=True, **kwargs)

    if name == "leakyrelu":
        return nn.LeakyReLU(inplace=True, **kwargs)

    if name == "tanh":
        return nn.Tanh()

    if name == "sigmoid":
        return nn.Sigmoid()

    raise ValueError("Unknown activation: '%s'" % name)


def get_norm_layer(norm_type="instance"):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "none":
        norm_layer = lambda _features: Identity()
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)

    return norm_layer


def get_norm_layer(norm, features):
    name, kwargs = extract_name_kwargs(norm)

    if name is None:
        return nn.Identity(**kwargs)

    if name == "layer":
        return nn.LayerNorm((features,), **kwargs)

    if name == "batch":
        return nn.BatchNorm2d(features, **kwargs)

    if name == "instance":
        return nn.InstanceNorm2d(features, **kwargs)

    raise ValueError("Unknown Layer: '%s'" % name)


def get_downsample_x2_conv2_layer(features, **kwargs):
    return (nn.Conv2d(features, features, kernel_size=2, stride=2, **kwargs), features)


def get_downsample_x2_conv3_layer(features, **kwargs):
    return (
        nn.Conv2d(features, features, kernel_size=3, stride=2, padding=1, **kwargs),
        features,
    )


def get_downsample_x2_pixelshuffle_layer(features, **kwargs):
    out_features = 4 * features
    return (nn.PixelUnshuffle(downscale_factor=2, **kwargs), out_features)


def get_downsample_x2_pixelshuffle_conv_layer(features, **kwargs):
    out_features = features * 4

    layer = nn.Sequential(
        nn.PixelUnshuffle(downscale_factor=2, **kwargs),
        nn.Conv2d(out_features, out_features, kernel_size=3, padding=1),
    )

    return (layer, out_features)


def get_downsample_x2_layer(layer, features):
    name, kwargs = extract_name_kwargs(layer)

    if name == "conv":
        return get_downsample_x2_conv2_layer(features, **kwargs)

    if name == "conv3":
        return get_downsample_x2_conv3_layer(features, **kwargs)

    if name == "avgpool":
        return (nn.AvgPool2d(kernel_size=2, stride=2, **kwargs), features)

    if name == "maxpool":
        return (nn.MaxPool2d(kernel_size=2, stride=2, **kwargs), features)

    if name == "pixel-unshuffle":
        return get_downsample_x2_pixelshuffle_layer(features, **kwargs)

    if name == "pixel-unshuffle-conv":
        return get_downsample_x2_pixelshuffle_conv_layer(features, **kwargs)

    raise ValueError("Unknown Downsample Layer: '%s'" % name)


def get_upsample_x2_layer(layer, features):
    name, kwargs = extract_name_kwargs(layer)

    if name == "deconv":
        return get_upsample_x2_deconv2_layer(features, **kwargs)

    if name == "upsample":
        return (nn.Upsample(scale_factor=2, **kwargs), features)

    if name == "upsample-conv":
        return get_upsample_x2_upconv_layer(features, **kwargs)

    if name == "pixel-shuffle":
        return (nn.PixelShuffle(upscale_factor=2, **kwargs), features // 4)

    if name == "pixel-shuffle-conv":
        return get_upsample_x2_pixelshuffle_conv_layer(features, **kwargs)

    raise ValueError("Unknown Upsample Layer: '%s'" % name)


def get_upsample_x2_deconv2_layer(features, **kwargs):
    return (
        nn.ConvTranspose2d(features, features, kernel_size=2, stride=2, **kwargs),
        features,
    )


def get_upsample_x2_upconv_layer(features, **kwargs):
    layer = nn.Sequential(
        nn.Upsample(scale_factor=2, **kwargs),
        nn.Conv2d(features, features, kernel_size=3, padding=1),
    )

    return (layer, features)


def get_upsample_x2_pixelshuffle_conv_layer(features, **kwargs):
    out_features = features // 4

    layer = nn.Sequential(
        nn.PixelShuffle(upscale_factor=2, **kwargs),
        nn.Conv2d(out_features, out_features, kernel_size=3, padding=1),
    )

    return (layer, out_features)
