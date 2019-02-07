from os import listdir
import numpy as np
import cv2
import random
import scipy.ndimage
import scipy.misc

__name__ = "training_data_frame"


class DataObject:
    # These are variables to prevent adding the same feature twice accidently.
    ROTATE = False
    SCALE = False
    BLUR = False
    NOISE = False

    def __init__(self, image):
        self.image_arr = image

    def get_array(self, shape=(6,)):
        return self.image_arr.flatten()

    def set_parent_features(self, parent_obj):
        self.ROTATE = parent_obj.ROTATE
        self.SCALE = parent_obj.SCALE
        self.BLUR = parent_obj.BLUR
        self.NOISE = parent_obj.NOISE


class TrainingDataFrame:

    data = {}
    letters = []
    DEFAULT_COLOR = 255.0

    # If images are white on black, pass false as a first argument, please.
    def __init__(self, black_on_white=True, root_dir="./data/ასოები/", height=25, width=25):
        self.HEIGHT = height
        self.WIDTH = width
        self.add_data(root_dir, black_on_white)

    # Taking parent (children of root_dir) folder names as labels, they should be only 1 letter long.
    # Data should be in labeled letter folders.
    # If images are white on black, pass false as a second argument, please.
    def add_data(self, root_dir, black_on_white=True):
        for letter in listdir(root_dir):
            if len(letter) > 1:
                continue
            for image_name in listdir(root_dir + letter):
                img = cv2.imread(root_dir + letter + "/" + image_name, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print("wrong image path")
                else:
                    if not black_on_white:
                        img = 255 - img
                        self.DEFAULT_COLOR = 0.0
                    resized_img = cv2.resize(img, dsize=(self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_CUBIC)
                    if letter not in self.data:
                        self.data[letter] = []
                        self.letters.append(letter)
                    self.data[letter].append(DataObject(resized_img))

    # Rotate alphas are angles.
    def add_rotate_f(self, rotate_alphas=(-20, -10, -5, 5, 10, 20)):
        rotate_alphas = list(set(rotate_alphas))
        rotate_alphas = [i for i in rotate_alphas if i % 360 != 0]  # removes angles which are useless
        if len(rotate_alphas) == 0:
            return
        for letter in self.letters:
            appendix = []
            for sample in self.data[letter]:
                if not sample.ROTATE:
                    sample.ROTATE = True
                    for angle in rotate_alphas:
                        new_sample = scipy.ndimage.interpolation.rotate(sample.get_array(), angle,
                                                                        mode='constant',
                                                                        cval=self.DEFAULT_COLOR,
                                                                        reshape=False)
                        new_dataobject = DataObject(new_sample)
                        new_dataobject.set_parent_features(sample)  # To prevent accidently using same feature twice.
                        appendix.append(new_dataobject)
            self.data[letter].extend(appendix)

    # Scale alphas are pixels to add edges (then resize to original size).
    # Warning: alphas that are bigger than 3 or smaller than -3 . passing them would cause an error.
    def add_scale_f(self, scale_alphas=(3, 2, 1, -1, -2, -3)):
        scale_alphas = list(set([int(i) for i in scale_alphas]))
        if 0 in scale_alphas:
            scale_alphas.remove(0)
        if len(scale_alphas) == 0:
            return
        for alpha in scale_alphas:
            assert -4 <= alpha <= 4
            if not -4 <= alpha <= 4:
                print(str(alpha) + " is forbidden, please pass correct scale alphas")
                return
        for letter in self.letters:
            appendix = []
            for sample in self.data[letter]:
                if not sample.SCALE:
                    sample.SCALE = True
                    for pixels in scale_alphas:
                        if pixels > 0:
                            new_sample = np.c_[np.full((self.HEIGHT + 2 * pixels, pixels), self.DEFAULT_COLOR),
                                               np.r_[np.full((pixels, self.WIDTH), self.DEFAULT_COLOR),
                                                     sample.get_array(),
                                                     np.full((pixels, self.WIDTH), self.DEFAULT_COLOR)],
                                               np.full((self.HEIGHT + 2 * pixels, pixels), self.DEFAULT_COLOR)]
                        else:
                            pixels *= -1
                            new_sample = sample.get_array()[pixels:-pixels, pixels:-pixels]
                        new_sample = cv2.resize(new_sample, dsize=(self.WIDTH, self.HEIGHT),
                                                interpolation=cv2.INTER_CUBIC)
                        new_dataobject = DataObject(new_sample)
                        new_dataobject.set_parent_features(sample)  # To prevent accidently using same feature twice.
                        appendix.append(new_dataobject)
            self.data[letter].extend(appendix)

    # Sigmas are values for blur coefficient. How much pixels should be interpolated to neighbour pixels.
    # Please keep values between 0 < sigma < 1.
    def add_blur_f(self, sigmas=(.1, .5)):
        sigmas = list(set(sigmas))
        sigmas = [i for i in sigmas if 0 < i < 1]  # removes values which are forbidden
        if len(sigmas) == 0:
            return
        for letter in self.letters:
            appendix = []
            for sample in self.data[letter]:
                if not sample.BLUR:
                    sample.BLUR = True
                    for sigma in sigmas:
                        new_sample = scipy.ndimage.gaussian_filter(sample.get_array(), sigma=sigma)
                        new_dataobject = DataObject(new_sample)
                        new_dataobject.set_parent_features(sample)  # To prevent accidently using same feature twice.
                        appendix.append(new_dataobject)
            self.data[letter].extend(appendix)

    # noise is maximum value added or decreased(max.:100), dots are how many dots are changed.
    def add_noise_f(self, noise=20, dots=10):
        if dots < 1 or 0 < noise < 100:
            return
        for letter in self.letters:
            appendix = []
            for sample in self.data[letter]:
                if not sample.NOISE:
                    sample.NOISE = True
                    new_sample = np.copy(sample.get_array())
                    for _ in range(dots):
                        x = random.randint(0, self.WIDTH - 1)
                        y = random.randint(0, self.HEIGHT - 1)
                        if new_sample[y][x] > 200:
                            noise *= -1
                        elif new_sample[y][x] > 50:
                            noise = random.randint(-noise, noise)
                        new_sample[y][x] = new_sample[y][x] + noise
                    new_dataobject = DataObject(new_sample)
                    new_dataobject.set_parent_features(sample)  # To prevent accidently using same feature twice.
                    appendix.append(new_dataobject)
            self.data[letter].extend(appendix)

    def get_random(self, letter):
        return random.choice(self.data[letter])

    def get_letter_list(self, letter):
        return self.data[letter]

    def get_letters(self):
        return self.letters

    def describe(self):
        print("data contains " + str(len(self.letters)) + "letters, ")
        total = 0
        for letter in self.letters:
            amount = len(self.data[letter])
            total += amount
            print(str(amount) + " - " + letter + "'s.")
        print("\nTOTAL: " + str(total) + " letters.")

