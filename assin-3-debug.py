#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[2]:


cp /kaggle/input/200-bird-species-with-11788-images/CUB_200_2011.tgz /kaggle/working/


# In[3]:


get_ipython().system('tar -xvzf CUB_200_2011.tgz')


# Pretty well balanced data set so we will use accuracy as our metric:

# In[4]:


# extract the required 7 classes
# classes assigned to us were 020, 128, 131, 95, 135, 41, 114
assigned_classes = ["020", "128", "131", "095", "135", "041", "114"]
for i in os.listdir("/kaggle/working/CUB_200_2011/images"):
    if i[:3] in assigned_classes:
        total_images = 0
        for j in os.listdir("/kaggle/working/CUB_200_2011/images/"+i):
            total_images += 1
        print(total_images)


# In[5]:


# maps filename to imageid
image_map = dict()
with open("/kaggle/working/CUB_200_2011/images.txt", "r") as file:
    for line in file:
        line = line.strip().split()
        image_map[line[1].split("/")[1]] = int(line[0])


# In[6]:


# maps imageid to bounding box coordinates
boundingbox_map = dict()
with open("/kaggle/working/CUB_200_2011/bounding_boxes.txt", "r") as file:
    for line in file:
        line = [int(float(l)) for l in line.split()]
        boundingbox_map[line[0]] = tuple(line[1:])


# Data preprocessing:

# In[7]:


X_data = []
y_data = []
class_val = 0
assigned_classes = ["020", "128", "131", "095", "135", "041", "114"]
for i in os.listdir("/kaggle/working/CUB_200_2011/images"):
    if i[:3] in assigned_classes:
        for j in os.listdir("/kaggle/working/CUB_200_2011/images/"+i):
            img = cv2.imread("/kaggle/working/CUB_200_2011/images/"+i+"/"+j)
            b = boundingbox_map[image_map[j]]
            img = img[b[1]:b[1]+b[3],b[0]:b[0]+b[2]]
            img = cv2.resize(img, (224,224), interpolation = cv2.INTER_NEAREST)
            X_data.append(img)
            y_data.append(class_val)
        class_val += 1
X_data = np.array(X_data)
y_data = np.array(y_data)


# In[8]:


# plot some images
plt.imshow(X_data[100])
plt.figure()
plt.imshow(X_data[230])
plt.figure()
plt.imshow(X_data[300])


# In[9]:


# do a train and test split
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size = 0.2)
X_train = X_train / 255
X_val = X_val / 255


# Here, we have defined a custom keras netvlad layer as defined in the paper:

# In[10]:


# define the neural network
reg = 1e-3
inputs = tf.keras.Input(shape = (224, 224, 3))
x = tf.keras.layers.Conv2D(filters = 4, kernel_size = (3,3), strides = (1,1), padding = "valid", activation = "relu", kernel_regularizer = tf.keras.regularizers.l2(reg), bias_regularizer = tf.keras.regularizers.l2(reg))(inputs)
x = tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = (2,2), padding = "valid")(x)
x = tf.keras.layers.Conv2D(filters = 16, kernel_size = (3,3), strides = (1,1), padding = "valid", activation = "relu", kernel_regularizer = tf.keras.regularizers.l2(reg), bias_regularizer = tf.keras.regularizers.l2(reg))(x)
x = tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = (2,2), padding = "valid")(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(5000, activation = "relu", kernel_regularizer = tf.keras.regularizers.l2(reg), bias_regularizer = tf.keras.regularizers.l2(reg))(x)
outputs = tf.keras.layers.Dense(7)(x)
model = tf.keras.Model(inputs = inputs, outputs = outputs, name = "vanilla")
model.compile(optimizer= tf.keras.optimizers.Adam(lr = 5e-5),
              loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs = 500, batch_size = 256, validation_data = (X_val, y_val),  verbose = 0)


# In[11]:


plt.figure(figsize = (12,7))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy data set 2')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
print(reg, max(history.history['val_accuracy']))


# In[12]:


plt.figure(figsize = (12,7))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss data set 2')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')


# In[13]:


# define custom keras netVLAD layer as descibed in the paper
class netvlad(tf.keras.layers.Layer):
    # takes input the number of clusters
    def __init__(self, num_k = 3):
        # inherits from the keras layer class
        super(netvlad, self).__init__()
        self.k = num_k
        
    def build(self, input_shape):
        # input shape is batch_size * height * weight * dimension
        # output shape is batch_size * cluster_size * dimension
        self.dimension = input_shape[-1]
        self.cluster_centers = self.add_weight(shape = (1,1,1,self.dimension,self.k), initializer = "random_normal", trainable = True, dtype = "float32")
        # define a conv2d layer and build it to later use it in our custom layer
        self.conv_layer = tf.keras.layers.Conv2D(filters = self.k, kernel_size = 1, strides = (1,1), use_bias = False)
        self.conv_layer.build(input_shape)
        super(netvlad, self).build(input_shape)
        
    def call(self, inputs):
        # as given in the netvlad paper use the conv layer first
        cv = self.conv_layer(inputs)
        # then pass through the softmax layer
        sm = tf.nn.softmax(cv)
        # expand the dims inorder to do vector processing
        sm = tf.expand_dims(sm, -2)
        # vlad operation as defined in paper
        v_lad = tf.expand_dims(inputs, -1) + self.cluster_centers
        v_lad = sm * v_lad
        v_lad = tf.reduce_sum(v_lad, axis = [1,2])
        v_lad = tf.transpose(v_lad, perm = [0,2,1])
        # normalise column wise and then thoughout the matrix as defined in the paper
        v_lad = v_lad / tf.sqrt(tf.reduce_sum(v_lad ** 2, axis=-1, keepdims=True) + 1e-10)
        v_lad = tf.transpose(v_lad, perm=[0, 2, 1])
        v_lad = tf.keras.layers.Flatten()(v_lad)
        v_lad = v_lad / tf.sqrt(tf.reduce_sum(v_lad ** 2, axis=-1, keepdims=True) + 1e-10)
        return v_lad


# In[14]:


# define the neural network
inputs = tf.keras.Input(shape = (224, 224, 3))
x = tf.keras.layers.Conv2D(filters = 4, kernel_size = (3,3), strides = (1,1), padding = "valid",)(inputs)
x = tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = (2,2), padding = "valid")(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Conv2D(filters = 16, kernel_size = (3,3), strides = (1,1), padding = "valid",)(x)
x = tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = (2,2), padding = "valid")(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = netvlad(4)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(20)(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(7)(x)
model = tf.keras.Model(inputs = inputs, outputs = outputs, name = "NetVLAD_implementation")
model.compile(optimizer= tf.keras.optimizers.Adam(lr = 1e-3),
              loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs = 1000, batch_size = 256, validation_data = (X_val, y_val),  verbose = 0)


# In[15]:


model.summary()


# In[16]:


plt.figure(figsize = (12,7))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy data set 2')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
print(max(history.history['val_accuracy']))


# In[17]:


plt.figure(figsize = (12,7))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss data set 2')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')


# In[ ]:




