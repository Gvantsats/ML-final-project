import pickle
import cv2
import os
import numpy as np


def sigmoid(s):
    return 1.0 / (np.exp(-s) + 1.0)


# Forward propagation
def forward(data, weights, biases):
    data = data.reshape(data.shape[0], 1)
    curr = data
    for i in range(len(biases)):
        bias = biases[i]
        weight = weights[i]
        mult = np.dot(weight, curr)
        curr = sigmoid(mult + bias)

    return curr


# saving network information from model
filename = "7_model.sav"

with open(filename, 'rb') as file:
    net_info = pickle.load(file)
    weights = net_info["weights"]
    biases = net_info["biases"]

label_names = ['უ', 'ყ', 'მ', 'შ', 'ძ', 'წ', 'ს', 'ხ', 'ლ', 'ჩ' , '-']
def classify(data , weights , biases):
        ans = forward(data, weights, biases)
        res = [0] * len(ans)
        print("ans ", ans)
        ind = -1
        for i in range(len(ans)):
            if ans[i] > 0.5:
                res[i] = 1
                ind = i
            else:
                res[i] = 0
        print(res)
        if (sum(res) > 1):
            return '-'
        return label_names[ind]

root_folder = "./data/ასოები/testing_data/"
data_list = []
for image_name in os.listdir(root_folder):
    print(image_name)
    img = cv2.imread(root_folder + image_name, cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(img, dsize=(25, 25), interpolation=cv2.INTER_CUBIC)
    reshaped = resized.reshape((625, 1))
    data_list.append((reshaped, image_name))

classified_data = []
for data, img_name in data_list:
    classified_label = classify(data, weights, biases)
    classified_data.append((img_name,classified_label))

print(classified_data)