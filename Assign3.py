#!/usr/bin/env python
# coding: utf-8

# In[110]:


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf


# In[87]:


shelf_images = 'C:/Users/Admin/Documents/Sem8/DL/Assign3/CUB_200_2011/images/'


# In[88]:


train_data = pd.read_csv('C:/Users/Admin/Documents/Sem8/DL/Assign3/train_data_team43.csv')
val_data = pd.read_csv('C:/Users/Admin/Documents/Sem8/DL/Assign3/val_data_team43.csv')
test_data = pd.read_csv('C:/Users/Admin/Documents/Sem8/DL/Assign3/test_data_team43.csv')


# In[89]:


num_classes = 200
SHAPE_WIDTH = 224
SHAPE_HEIGHT = 224
def resize_pack(pack):
    fx_ratio = SHAPE_WIDTH / pack.shape[1]
    fy_ratio = SHAPE_HEIGHT / pack.shape[0]    
    pack = cv2.resize(pack, (0, 0), fx=fx_ratio, fy=fy_ratio)
    return pack[0:SHAPE_HEIGHT, 0:SHAPE_WIDTH]


# In[90]:


# x - image, y - class, f - is_train flag
x, y, f = [], [], []
for file, is_train in train_data[['file', 'is_train']].values:
    photo_rects = train_data[train_data.file == file]
    rects_data = photo_rects[['label', 'xmin', 'ymin', 'xmax', 'ymax']]
    im = cv2.imread(f'{shelf_images}{file}')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    for category, xmin, ymin, xmax, ymax in rects_data.values:
        pack = resize_pack(np.array(im[ymin:ymax, xmin:xmax]))
        x.append(pack)
        f.append(is_train)
        y.append(category)
x_train = np.array(x)
y_train = np.array(y)


# In[91]:


x, y, f = [], [], []
for file, is_train in val_data[['file', 'is_train']].values:
    photo_rects = val_data[val_data.file == file]
    rects_data = photo_rects[['label', 'xmin', 'ymin', 'xmax', 'ymax']]
    im = cv2.imread(f'{shelf_images}{file}')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    for category, xmin, ymin, xmax, ymax in rects_data.values:
        pack = resize_pack(np.array(im[ymin:ymax, xmin:xmax]))
        x.append(pack)
        f.append(is_train)
        y.append(category)
x_val = np.array(x)
y_val = np.array(y)


# In[92]:


x, y, f = [], [], []
for file, is_train in test_data[['file', 'is_train']].values:
    photo_rects = test_data[test_data.file == file]
    rects_data = photo_rects[['label', 'xmin', 'ymin', 'xmax', 'ymax']]
    im = cv2.imread(f'{shelf_images}{file}')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    for category, xmin, ymin, xmax, ymax in rects_data.values:
        pack = resize_pack(np.array(im[ymin:ymax, xmin:xmax]))
        x.append(pack)
        f.append(is_train)
        y.append(category)
x_test = np.array(x)
y_test = np.array(y)


# In[93]:


# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_val = keras.utils.to_categorical(y_val, num_classes)
from keras.utils import to_categorical
y_train = to_categorical(y_train,200)
y_val = to_categorical(y_val,200)
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_train /= 255
x_val /= 255


# In[168]:


from keras.regularizers import l1,l2
model = Sequential()
model.add(Conv2D(4, kernel_size=(3, 3),kernel_regularizer=l1(1e-3),activation='relu',input_shape=(224,224,3)))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu',kernel_regularizer=l1(1e-3)))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Flatten())
# model.add(Dense(256, activation = "relu"))
model.add(Dense(200, activation = "softmax"))


# In[169]:


model.summary()


# In[170]:


y_train.shape


# In[171]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy','accuracy'])
batch_size = 16

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20)


# In[172]:


import matplotlib.pyplot as plt
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1,21)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[173]:


pred = model.predict(x_test)


# In[184]:


pred = np.argmax(pred, axis=1)
n_correct = np.sum(np.equal(pred, y_test).astype(int))
total = float(len(pred)) 
print("Test Accuracy:", round(n_correct / total, 3))


# In[179]:


pred = model.predict(x_train)


# In[180]:


pred = np.argmax(pred, axis=1)
n_correct = np.sum(np.equal(pred, np.argmax(y_train, axis=1)).astype(int))
total = float(len(pred)) 
print("Train Accuracy:", round(n_correct / total, 3))


# In[178]:


pred = model.predict(x_val)


# In[182]:


pred = np.argmax(pred, axis=1)
n_correct = np.sum(np.equal(pred, np.argmax(y_val, axis=1)).astype(int))
total = float(len(pred)) 
print("Validation Accuracy:", round(n_correct / total, 3))


# In[ ]:




