from os import listdir, rename
import numpy as np
from PIL import Image

ROOT_DIR = "./data/ასოები/"

for letter in listdir(ROOT_DIR):
    if len(letter) > 1:
        continue
    for image_name in listdir(ROOT_DIR+letter):
        print(ROOT_DIR + letter + "/" + image_name)
        print("./data/ასოები/" + letter + "/" + image_name.strip(".jpg") + "_flipped.jpg")
        rename(ROOT_DIR + letter + "/" + image_name,
             ROOT_DIR + letter + "/" + letter + "_" + image_name)
