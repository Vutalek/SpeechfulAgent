import numpy as np
from PIL import Image, ImageFont, ImageDraw

from speechfulagent.dataclasses import Experience


def make_image(exp: Experience, frame: np.ndarray):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 9)
    draw.text((0, 0), str(exp).replace("Experience", ""), (0, 0, 0), font=font)
    return img