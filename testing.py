import torch
import clip
from PIL import Image
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("test_folder/test.png")).unsqueeze(0).to(device)


with torch.no_grad():
    for _ in range(10):
        start_time = time.time()
        image = preprocess(Image.open("test_folder/test.png")).unsqueeze(0).to(device)
        image_features = model.encode_image(image)
        print(image_features.shape)
        print("Time:", time.time()-start_time)
