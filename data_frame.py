from os import listdir
import numpy as np
import cv2
import random

__name__ = "training_data_frame"


class DataObject:

    def __init__(self, image):
        self.image_arr = image

    def get_array(self):
        return self.image_arr


class TrainingDataFrame:

    data = {}

    ROOT_DIR = "./data/ასოები/"

    # If images are white on black, pass false as a first argument, please.
    def __init__(self, black_on_white=True):
        for letter in listdir(self.ROOT_DIR):
            if len(letter) > 1:
                continue
            for image_name in listdir(self.ROOT_DIR + letter):
                img = cv2.imread(self.ROOT_DIR + letter + "/" + image_name)
                if img is None:
                    print("wrong image path")
                else:
                    if not black_on_white:
                        img = 255 - img
                    resized_img = cv2.resize(img, dsize=(25, 25), interpolation=cv2.INTER_CUBIC)
                    if letter not in self.data:
                        self.data[letter] = []
                    self.data[letter].append(DataObject(resized_img))

    def get_random(self, letter):
        return random.choice(self.data[letter])

    def get_letter_list(self, letter):
        return self.data[letter]

    def get_letters(self):
        return list(self.data.keys())

    def describe(self):
        print("data contains " + str(len(self.data.keys())) + "letters, ")
        total = 0
        for letter in self.data.keys():
            amount = len(self.data[letter])
            total += amount
            print(str(amount) + " - " + letter + "'s.")
        print("\nTOTAL: " + str(total) + " letters.")
