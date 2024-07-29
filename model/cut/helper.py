import os
from dotenv import load_dotenv, find_dotenv
from model.common import load_generator


load_dotenv(find_dotenv(), override=True)


def load_b_to_a_generator():
    MODEL_PATH = os.getenv("CUT_STATUE_TO_HUMAN_MODEL")

    return load_generator(MODEL_PATH, "cut")
