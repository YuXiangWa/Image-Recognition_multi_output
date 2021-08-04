# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 22:29:17 2021

@author: user
"""
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.utils import plot_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from model_name.fashionnet import FashionNet
from my_utils import utils_paths
import matplotlib.pyplot as plt
import numpy as np

import random
import pickle
import cv2
import os

# 設置參數

print("[INFO] 加載數據...")
EPOCHS = 50
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)

imagePaths = sorted(list(utils_paths.list_images("dataset")))
random.seed(42)
random.shuffle(imagePaths)

data = []
categoryLabels = []
colorLabels = []


for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = img_to_array(image)
	data.append(image)

	(color, cat) = imagePath.split(os.path.sep)[-2].split("_")
	categoryLabels.append(cat)
	colorLabels.append(color)

data = np.array(data, dtype="float") / 255.0

categoryLabels = np.array(categoryLabels)
colorLabels = np.array(colorLabels)

categoryLB = LabelBinarizer()
colorLB = LabelBinarizer()
categoryLabels = categoryLB.fit_transform(categoryLabels)
colorLabels = colorLB.fit_transform(colorLabels)

# 數據集切分
split = train_test_split(data, categoryLabels, colorLabels,
	test_size=0.2, random_state=42)
(trainX, testX, trainCategoryY, testCategoryY,
	trainColorY, testColorY) = split

# 創建網路
model = FashionNet.build(96, 96,
	numCategories=len(categoryLB.classes_),
	numColors=len(colorLB.classes_),
	finalAct="softmax")

# 指定loss
losses = {
	"category_output": "categorical_crossentropy",
	"color_output": "categorical_crossentropy",
}
lossWeights = {"category_output": 1.0, "color_output": 1.0}

# 訓練模型
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
	metrics=["accuracy"])
print("[INFO] 訓練網路...")
H = model.fit(trainX,
	{"category_output": trainCategoryY, "color_output": trainColorY},
	validation_data=(testX,
		{"category_output": testCategoryY, "color_output": testColorY}),
	epochs=EPOCHS,
	verbose=1)
# 保存模型
model.save("outputs\model.model")

# 保存標籤
f = open("outputs\category_lb.pickle", "wb")
f.write(pickle.dumps(categoryLB))
f.close()

f = open("outputs\color_lb.pickle", "wb")
f.write(pickle.dumps(colorLB))
f.close()

lossNames = ["loss", "category_output_loss", "color_output_loss"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))

for (i, l) in enumerate(lossNames):
	title = "Loss for {}".format(l) if l != "loss" else "Total loss"
	ax[i].set_title(title)
	ax[i].set_xlabel("Epoch #")
	ax[i].set_ylabel("Loss")
	ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
	ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l],
		label="val_" + l)
	ax[i].legend()


plt.tight_layout()
plt.savefig("output_losses.png")
plt.close()


accuracyNames = ["category_output_acc", "color_output_acc"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(2, 1, figsize=(8, 8))


for (i, l) in enumerate(accuracyNames):
	ax[i].set_title("Accuracy for {}".format(l))
	ax[i].set_xlabel("Epoch #")
	ax[i].set_ylabel("Accuracy")
	ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
	ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l],
		label="val_" + l)
	ax[i].legend()


plt.tight_layout()
plt.savefig("output_accs.png")
plt.close()
plot_model(model, to_file='model.png',show_shapes=True)
