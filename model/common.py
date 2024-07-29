import torch
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from torch import Tensor
from model.cut.models import SinCUTModel, CUTModel
from model.uvcgan2.networks import VitModNetGenerator
from types import SimpleNamespace

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_generator(MODEL_PATH, model_type):

    if model_type == "uvcgan2":
        model = VitModNetGenerator()

        current_model_dict = model.state_dict()
        loaded_state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        new_state_dict = {
            k: v if v.size() == current_model_dict[k].size() else current_model_dict[k]
            for k, v in zip(current_model_dict.keys(), loaded_state_dict.values())
        }
        model.load_state_dict(new_state_dict, strict=False)
        model.to(DEVICE)

    elif model_type == "cut":
        opt = SimpleNamespace(
            **{
                "gpu_ids": "0",
                "isTrain": False,
                "checkpoints_dir": "/home/erenaspire7/repos/honours-project/rest-api/checkpoints",
                "name": "statue2human_CUT",
                "preprocess": "resize_and_crop",
                "nce_layers": "0,4,8,12,16",
                "nce_idt": False,
                "input_nc": 3,
                "output_nc": 3,
                "ngf": 64,
                "ndf": 64,
                "netD": "basic",
                "netG": "resnet_9blocks",
                "normG": "instance",
                "normD": "instance",
                "no_dropout": True,
                "init_type": "xavier",
                "init_gain": 0.02,
                "no_antialias": False,
                "no_antialias_up": False,
                "netF": "mlp_sample",
                "netD": "basic",
                "netF_nc": 256,
            }
        )

        model = SinCUTModel(opt)

        model.load_networks("130")
        model.print_networks(True)

    else:
        raise Exception()

    model.eval()

    return model


def convert_tensor_to_image(tensor: Tensor):
    tensor = tensor.squeeze(0).cpu().detach().numpy()

    # Denormalize if necessary (assuming the tensor was normalized to [0, 1])
    tensor = (tensor * 255).astype(np.uint8)

    # Change from (C, H, W) to (H, W, C) format
    tensor = tensor.transpose(1, 2, 0)

    img = Image.fromarray(tensor)

    img.save("my.png")

    return img


def convert_to_pil_image(source: str, type: str):
    if type == "base64":
        return Image.open(BytesIO(base64.b64decode(source)))
    else:
        return Image.open(source)


def generate_image(source: str, model, type="base64"):
    img = convert_to_pil_image(source, type)

    img = img.resize([256, 256], Image.LANCZOS)

    tensor_img = image_to_tensor(img)

    output = model(tensor_img)

    convert_tensor_to_image(output)


def image_to_tensor(img: Image):
    tensor = torch.tensor(np.array(img)).permute(2, 0, 1)  # Change to (C, H, W) format
    tensor = tensor.float()
    tensor = tensor.to(DEVICE)
    tensor = tensor.unsqueeze(0)
    tensor = tensor / 255.0
    return tensor
