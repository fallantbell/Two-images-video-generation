import torch
import os
import clip
from PIL import Image
import time

def center_crop_and_resize(frame: Image, height: int, width: int) -> Image:
    #* 去除inf nature 黑邊
    frame = frame.crop((0,(1/5)*frame.height,frame.width,(4/5)*frame.height))

    # Measures by what factor height and width are larger/smaller than desired.
    height_scale = frame.height / height
    width_scale = frame.width / width

    # Center crops whichever dimension has a greater scale factor.
    if height_scale > width_scale:
        crop_height = height * width_scale
        y0 = (frame.height - crop_height) // 2
        y1 = y0 + crop_height
        frame = frame.crop((0, y0, frame.width, y1))

    elif width_scale > height_scale:
        crop_width = width * height_scale
        x0 = (frame.width - crop_width) // 2
        x1 = x0 + crop_width
        frame = frame.crop((x0, 0, x1, frame.height))

    # Resizes to desired height and width.
    frame = frame.resize((width, height), Image.LANCZOS)
    return frame


dataset_dir = "../../../disk2/icchiu/acid_dataset/train"

tmp = 0
for dir in os.listdir(dataset_dir):
    dir_name = dataset_dir+"/"+dir

    for img in os.listdir(dir_name):
        img_path = dir_name+"/"+img
        frame = Image.open(img_path)
        frame = center_crop_and_resize(frame,36,64)
        frame.save(f"img_folder/{tmp}.png")
        break

    tmp+=1