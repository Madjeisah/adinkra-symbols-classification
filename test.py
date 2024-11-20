# set the matplotlib backend so figures can be saved in the background
import matplotlib
# matplotlib.use("Agg")
# import the necessary packages
from models.resnet import ResNet
from models.vgg import VGGNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

from assets import config


# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print(" ")
print(" ")
print("[INFO] loading image paths...")
imagePaths = list(paths.list_images(config.TEST_DATASET_PATH))

data = []
labels = []
	
# loop over the train image paths
for imagePath in imagePaths:
	
	preprocess_threads=config.NUM_DEVICES
	
	# extract the class label from the filename, load the image
	label = imagePath.split(os.path.sep)[-2]
	
	# resize it to be a fixed 64x64 pixels, ignoring aspect ratio
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (32, 32))
		
	# update the data and labels lists, respectively
	data.append(image)
	labels.append(label)
print("[INFO] Train set done loading!")
print("[INFO]", len(data))
print(" ")

		

# convert the data into a NumPy array, then preprocess it by scaling
# all pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0

# encode the labels (which are currently strings) as integers and then
# one-hot encode them
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, 62)


# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.2, random_state=42)
	

# initialize the optimizer and model
print("[INFO] compiling model...")
#opt = tf.keras.optimizers.legacy.SGD(learning_rate=config.INIT_LR, momentum=0.9, decay=config.INIT_LR / config.EPOCHS)
opt = tf.keras.optimizers.SGD(learning_rate=config.INIT_LR, momentum=0.9, decay=config.INIT_LR / config.EPOCHS)
model = ResNet.build(32, 32, 3, 62, (2, 3, 4), (32, 64, 128, 256), reg=0.0001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
	
	
# train the network without aug
print(" ") 
print("[INFO] training network for {} epochs...".format(config.EPOCHS))
H = model.fit(trainX, trainY, batch_size=config.BS,
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // config.BS,
	epochs=config.EPOCHS, verbose=1)


# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(x=testX.astype("float32"), batch_size=config.BS)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=le.classes_))
	

# plot the training loss and accuracy
N = np.arange(0, config.EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(config.SAVE)
plt.show()	
	



















