import os, sys

sys.path.insert(0, os.getcwd())

from model.uvcgan2.helper import (
    load_a_to_b_generator,
    load_b_to_a_generator,
)
from model.common import generate_image

human_to_statue_model = load_a_to_b_generator()
human_image_path = "/home/erenaspire7/repos/honours-project/rest-api/tests/human_2.jpg"

statue_to_human_model = load_b_to_a_generator()
statue_image_path = (
    "/home/erenaspire7/repos/honours-project/rest-api/tests/statue_3.png"
)

# transformed = generate_image(human_image_path, human_to_statue_model, "image-path")
generate_image(statue_image_path, statue_to_human_model, "image-path")
