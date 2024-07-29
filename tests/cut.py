import os, sys

sys.path.insert(0, os.getcwd())


from model.cut.helper import load_b_to_a_generator
from model.common import generate_image


statue_to_human_model = load_b_to_a_generator()


statue_image_path = (
    "/home/erenaspire7/repos/honours-project/rest-api/tests/statue_3.png"
)


print(statue_to_human_model.model_names)


model = getattr(statue_to_human_model, "netG")


generate_image(statue_image_path, model, "image-path")
