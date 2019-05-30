import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical 
from keras.preprocessing import image
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
from tqdm import tqdm #Used to display a progress bar

from PIL import Image
import os
import glob
import cv2
import re

import sys
from sys import getsizeof #to check size of variables, in bytes (Divide value by 1e+9 to check in Gb )
from PIL import Image
sys.modules['Image'] = Image

training_labels = pd.read_csv(r"E:\datasets\ShipOrVesselDetection\train\train.csv")

#Checking for the class:
training_labels[training_labels.columns[1]].value_counts()

'''
1    2120
5    1217
2    1167
3     916
4     832
'''

#Path variables
datapath = r"E:\datasets\ShipOrVesselDetection\train\images"

#The items and the number of items in the directory
items = glob.glob1(datapath, "*.*")
count_items = len(items)

#Checking the file extensions - FYI, there are only jpg's in here
extenstion_check = []
for i in items:
    ext = i.split(".", 1)[1]
    extenstion_check.append(ext)

pd.Series(extenstion_check).value_counts()
del extenstion_check

#There are 8932 images.
#The plan is to use the items in 'training_labels' to run the training module, then test with the 'test' one - It is a 70-30 split

#Checking sizes of every image
#We're doing this to know how much to resize the image by during the image normalization process (which will come later)
image_size_collection = []
for every_image in range(len(items)):
    size_of_images = cv2.imread(datapath + "\\" + items[every_image])
    image_size_collection.append(size_of_images)

#'image_size_collection' is a list that contain 8932 elements, each of which are the matrix-equivalent of the image.
#We iterate over each item to check its size

x_dimension = []
y_dimension = []
z_dimension = []
for i in range(len(image_size_collection)):
   x_dimension.append(image_size_collection[i].shape[0])
   y_dimension.append(image_size_collection[i].shape[1])
   z_dimension.append(image_size_collection[i].shape[2])

#Max and min of the x_dimension
xmax = max(x_dimension)
xmin = min(x_dimension)
xvar = statistics.variance(x_dimension)
xstddev = statistics.stdev(x_dimension)

#Max and min of the y_dimension
ymax = max(y_dimension)
ymin = min(y_dimension)
yvar = statistics.variance(y_dimension)
ystddev = statistics.stdev(y_dimension)

#They are all 3 dimentional images, because
pd.Series(z_dimension).value_counts() # = 3

#Checking if there are any images where x coordinate > y coordinates (portrait mode)
portrait_images = []
for i in range(len(image_size_collection)):
    if(image_size_collection[i].shape[0] > image_size_collection[i].shape[1]):
        portrait_images.append(image_size_collection[i])

#Gives you the number of image that are in portrait mode (resizing may cause some issues)
len(portrait_images)


#Training set
X = []
for i in range(len(training_labels.image)):
    img = cv2.imread(datapath + '\\' + training_labels.image[i])
    img = cv2.resize(img, (xmin, xmin), interpolation = cv2.INTER_AREA)
    X.append(img)

x_train = np.asarray(X) #Shape = (6252, 41, 41, 3)
y_train = training_labels.category

#Encoding the output variable
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_y = encoder.transform(y_train)
dummy_y = np_utils.to_categorical(encoded_y)

dummy_y.shape #(6252, 5)

#THE MODEL
model = Sequential()

#Input layer 
model.add(
        Conv2D(
                32,
                kernel_size = (3, 3),
                strides = (1, 1),
                activation = 'relu',
                input_shape = x_train.shape[1:]))
model.add(BatchNormalization())
model.add(Dropout(.25))

#Second convoluation layer
model.add(
    Conv2D(
        64, 
        (3, 3), 
        activation = 'relu'))

#Maxpooling layer
model.add(MaxPooling2D(pool_size = (2, 2)))

#Third convoluation layer
model.add(
    Conv2D(
        128, 
        (3, 3), 
        activation = 'relu'))

#Maxpooling layer
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())

#Hidden layers
#Initializations define the way to set the initial random weights of Keras layers.
model.add(Dense(50, activation = "tanh", kernel_initializer ='random_uniform', bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(Dropout(.25))
model.add(Dense(50, activation = "tanh", kernel_initializer ='random_uniform', bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(Dropout(.25))
model.add(Dense(50, activation = "tanh", kernel_initializer ='random_uniform', bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(Dropout(.25))
model.add(Dense(50, activation = "tanh", kernel_initializer ='random_uniform', bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(Dropout(.25))
model.add(Dense(50, activation = "tanh", kernel_initializer ='random_uniform', bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(Dropout(.25))
model.add(Dense(50, activation = "tanh", kernel_initializer ='random_uniform', bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(Dropout(.25))
model.add(Dense(50, activation = "tanh", kernel_initializer ='random_uniform', bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(Dropout(.25))
model.add(Dense(50, activation = "tanh", kernel_initializer ='random_uniform', bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(Dropout(.25))
#output layer
model.add(Dense(5, activation = 'sigmoid'))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss = "categorical_crossentropy", optimizer = sgd, metrics = ['accuracy'])

model.summary()

#Fitting the model
#But first
x_train = x_train.astype('float32')
x_train /= 255

model_fit = model.fit(x_train, dummy_y, batch_size = 100, epochs = 150, verbose = 1, validation_split = .2)

scores = model.evaluate(x_train, dummy_y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#Plot of Loss
plt.figure()
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('MODEL LOSS')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

#Plot of Accuray
plt.figure()
plt.plot(model.history.history['acc'])
plt.plot(model.history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

#Saving the mode:
model_json = model.to_json()
with open("model01.json", "w") as json_file:
    json_file.write(model_json)

#Serialize weights to HDF5
model.save_weights("model_weights01.h5")
print("Saved model to disk")

#The testing part
test = pd.read_csv(r"E:\datasets\ShipOrVesselDetection\test_ApKoW4T.csv")

#Converting the image to arrays and resizing (just as before)
test_set = []
for i in range(len(test)):
    test_img = cv2.imread(datapath + '\\' + test.image[i])
    test_img = cv2.resize(test_img, (xmin, xmin), interpolation = cv2.INTER_AREA)
    test_set.append(test_img)

test_data = np.array(test_set)
test_data.shape #(2680, 41, 41, 3)

y_hat = model.predict_classes(test_data)
y_hat += 1 #Because y_hat ranges between 0 and 4. Adding 1 to it ranges from 1 to 5, as per the problem statement


for i in range(len(test)):
    print(f"{test.image[i]} - {y_hat[i]}")
    
#Adding a new column to the test df, called 'predictions'
test["category"] = y_hat

#Importing df to csv
test.to_csv("Predictions01.csv", index = False)