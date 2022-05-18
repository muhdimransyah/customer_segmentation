# -*- coding: utf-8 -*-
"""
Created on Wed May 18 09:51:13 2022

@author: imran
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten

tp = os.path.join(os.getcwd(), 'train.csv')
test_path = os.path.join(os.getcwd(), 'new_customers.csv')
#MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'model.h5')
#LOG_PATH = os.path.join(os.getcwd(),'log')

# EDA 
# Step 1) Data Loading
(x_train, y_train), (x_test,y_test) = tp.load_data()

# Step 2) Data Intepretation
fig, axes = plt.subplots(ncols=5, sharex=(False), sharey=(True))

for i in range(5):
    axes[i].set_title(y_train[i])
    axes[i].imshow(x_train[i], cmap='gray')
    axes[i].get_xaxis().set_visible(False)
    axes[i].get_yaxis().set_visible(False)

plt.show()

enc = OneHotEncoder(sparse=False)
y_train = enc.fit_transform(np.expand_dims(y_train,axis=-1))
y_test = enc.transform(np.expand_dims(y_test,axis=-1))

nb_classes = 2 # categories

#%% Improve Model

# Step 1) Creating a Container
model = Sequential()

# Step 2) Placing items into Container
model.add(Input(shape=(28,28), name='input_Layer')) # 2-d
model.add(Flatten()) # to flatten the data
model.add(Dense(5, activation="sigmoid", name="hidden_layer_1"))
model.add(BatchNormalization())
model.add(Dense(10, activation="sigmoid", name="hidden_layer_2"))
model.add(BatchNormalization())
model.add(Dense(nb_classes, activation="softmax", name="output_layer"))
model.summary()

# Step 3) Wrap the Container
model.compile(optimizer=tf.keras.optimizers.Adam(), # optimizer
              loss=tf.keras.losses.CategoricalCrossentropy(), # losses
              metrics = tf.keras.metrics.Accuracy() # metrics
) 

# Step 4) Model Training
hist = model.fit(x_train,y_train, epochs=5, validation_data=(x_test,y_test))

hist.history.keys()
training_loss = hist.history['loss']
training_acc = hist.history['accuracy']
validation_loss = hist.history['val_loss']
validation_acc = hist.history['val_accuracy']  

# Model Performances
plt.figure()
plt.plot(training_loss)
plt.plot(validation_loss)
plt.title("Training Loss")
plt.xlabel('epoch')
plt.ylabel('Cross entropy loss')
plt.legend(['training loss','validation loss'])
plt.show()

plt.figure()
plt.plot(training_acc)
plt.plot(validation_acc)
plt.title("Training Accuracy")
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.legend(['training acc','validation acc'])
plt.show()

results = model.evaluate(x_test,y_test,batch_size=100) #loss


