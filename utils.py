import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import random

data_dir = './flowers'
data = []
img_size = 125
categories = ["daisy", "dandelion", "rose", "sunflower", "tulip"]


def create_data():
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            img_arr = cv2.imread(os.path.join(path, img))
            try:
                new_arr = cv2.resize(img_arr, (img_size, img_size))
            except cv2.error as e:
                pass
            cv2.waitKey()

            data.append([new_arr, class_num])


create_data()


def load_data():
    random.shuffle(data)
    X = []
    y = []

    for features, labels in data:
        X.append(features)
        y.append(labels)

    X = np.array(X).reshape(-1, img_size, img_size, 3)
    y = np.array(y)
    return X, y


(X, y) = load_data()
pickle_out = open('X.pickle', 'wb')

pickle.dump(X, pickle_out)

pickle_out_2 = open('y.pickle', 'wb')

pickle.dump(y, pickle_out_2)
