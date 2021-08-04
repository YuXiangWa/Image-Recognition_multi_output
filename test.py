# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:50:17 2021

@author: user
"""
from keras.models import load_model
import tensorflow as tf
import numpy as np
import pickle
import cv2
from my_utils import utils_paths
import matplotlib.pyplot as plt

# 加載網路

model = load_model("output\model.model",custom_objects={"tf": tf})
categoryLB = pickle.loads(open("output\category_lb.pickle", "rb").read())
colorLB = pickle.loads(open("output\color_lb.pickle", "rb").read())

imagePaths = sorted(list(utils_paths.list_images("test")))
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    output = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = np.expand_dims(image, axis=0)



    (categoryProba, colorProba) = model.predict(image)

# 取得預測值
    categoryIdx = categoryProba[0].argmax()
    colorIdx = colorProba[0].argmax()
    categoryLabel = categoryLB.classes_[categoryIdx]
    colorLabel = colorLB.classes_[colorIdx]

# 繪圖展示
    categoryText = "category: {} ({:.2f}%)".format(categoryLabel,
        categoryProba[0][categoryIdx] * 100)
    colorText = "color: {} ({:.2f}%)".format(colorLabel,
        colorProba[0][colorIdx] * 100)
    cv2.putText(output, categoryText, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 0), 2)
    cv2.putText(output, colorText, (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 0), 2)

# 打印結果
    print("[INFO] {}".format(categoryText))
    print("[INFO] {}".format(colorText))
    
    output = output[:,:,::-1]
    plt.figure(figsize=(7,7))
    plt.imshow(output)
    plt.show()