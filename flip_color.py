from os import listdir, path, makedirs
import numpy as np
from PIL import Image

ROOT_DIR = "./data/georgial-language-ocr-data-master/"

for letter in listdir(ROOT_DIR):
    if len(letter) > 1:
        continue
    if not path.exists("./data/flipped/" + letter):
        makedirs("./data/flipped/" + letter)
    for image_name in listdir(ROOT_DIR+letter):
        print(ROOT_DIR + letter + "/" + image_name)
        with Image.open(ROOT_DIR + letter + "/" + image_name) as image:
            im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
            im_arr = im_arr.reshape((image.size[1], image.size[0]))
            im_arr = 255 - im_arr
            img = Image.fromarray(im_arr, 'L')
            img.save("./data/flipped/" + letter + "/" + image_name)
