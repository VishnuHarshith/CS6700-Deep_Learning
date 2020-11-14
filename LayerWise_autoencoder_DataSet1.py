#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
import keras
import keras.backend as K
from keras.layers import Input, Convolution2D, Activation, MaxPooling2D,Dense, BatchNormalization, Dropout
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.models import Model
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler
from keras.layers.normalization import BatchNormalization
from keras import regularizers
print(keras.__version__)
import tensorflow as tf


# In[123]:


import os
import pandas as pd
import numpy as np
directory = os.fsencode("C:\\Users\\Admin\\Documents\\Sem8\\DL\\Data_Set_1(Colored_Images)\\input")
i = 4
dlist = []
dtar = []
data = np.array([])
temp = np.array([])
temp_tar = np.array([])
for folder in os.listdir(directory):
    print(folder,i)
    for file in os.listdir(folder):
        dtar.append(i)
        filename = os.fsdecode(file)
        filename = os.fsdecode(directory) + "\\" + os.fsdecode(folder) + "\\" + filename
        temp = np.loadtxt(filename, delimiter=" ")
        dlist.append(temp)
    i-=1


# In[124]:


data = np.stack(dlist,axis = 0)
targets = np.stack(dtar,axis = 0)
targets = targets.reshape(len(targets),1)
targets = np_utils.to_categorical(targets)


# In[125]:


target_arr = np.stack(dtar,axis = 0)
target_arr = target_arr.reshape(len(targets),1)


# In[126]:


from sklearn.utils import shuffle
import random
data, targets, target_arr = shuffle(data, targets, target_arr, random_state=0)


# In[127]:


N_train = 1331
training_inputs = data[0:N_train,:,:].astype('float32')
val_inputs = data[(N_train+1):-1,:,:].astype('float32')
training_inputs = training_inputs.reshape((len(training_inputs), np.prod(training_inputs.shape[1:])))
val_inputs = val_inputs.reshape((len(val_inputs), np.prod(val_inputs.shape[1:])))


# In[128]:


from sklearn.preprocessing import minmax_scale
training_inputs = minmax_scale(training_inputs, feature_range=(0,1), axis=1)
val_inputs = minmax_scale(val_inputs, feature_range=(0,1), axis=1)
training_targets = targets[0:N_train,:]
val_targets = targets[(N_train+1):-1,:]
t_targets = target_arr[0:N_train,:]
v_targets = target_arr[(N_train+1):-1,:]


# In[129]:


percent_noisy = 0.5
indices_tozero = np.random.choice(range(training_inputs.shape[0] * training_inputs.shape[1]), int(percent_noisy * training_inputs.shape[0] * training_inputs.shape[1]), replace = False)
training_inputs_noisy = training_inputs.copy()
np.put(training_inputs_noisy, indices_tozero, 0)
val_inputs_noisy = val_inputs.copy()
indices_tozero = np.random.choice(range(val_inputs.shape[0] * val_inputs.shape[1]), int(percent_noisy * val_inputs.shape[0] * val_inputs.shape[1]), replace = False)
np.put(val_inputs_noisy, indices_tozero, 0)


# In[130]:


input_img = Input(shape = (828, ))
encoded1 = Dense(1500,activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(input_img)
# encoded1_bn = BatchNormalization()(encoded1)
decoded1 = Dense(828)(encoded1)

autoencoder1 = Model(input = input_img, output = decoded1)
encoder1 = Model(input = input_img, output = encoded1)

# Layer 2
encoded1_input = Input(shape = (1500,))
# distorted_input2 = Dropout(.2)(encoded1_input)
encoded2 = Dense(1000, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(encoded1_input)
# encoded2_bn = BatchNormalization()(encoded2)
decoded2 = Dense(1500)(encoded2)

autoencoder2 = Model(input = encoded1_input, output = decoded2)
encoder2 = Model(input = encoded1_input, output = encoded2)

# Layer 3
encoded2_input = Input(shape = (1000,))
# distorted_input3 = Dropout(.3)(encoded2_input)
encoded3 = Dense(1000, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(encoded2_input)
# encoded3_bn = BatchNormalization()(encoded3)
decoded3 = Dense(1000)(encoded3)

autoencoder3 = Model(input = encoded2_input, output = decoded3)
encoder3 = Model(input = encoded2_input, output = encoded3)

# Deep Autoencoder
encoded1_da = Dense(1500,activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-3, l2=1e-3))(input_img)
# encoded1_da_bn = BatchNormalization()(encoded1_da)
encoded2_da = Dense(1000,activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(encoded1_da)
# encoded2_da_bn = BatchNormalization()(encoded2_da)
encoded3_da = Dense(1000,activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(encoded2_da)
# encoded3_da_bn = BatchNormalization()(encoded3_da)
# decoded3_da = Dense(400,activation='relu')(encoded3_da_bn)
# decoded2_da = Dense(1000,activation='relu')(decoded3_da)
# decoded1_da = Dense(828,activation='sigmoid')(decoded2_da)

deep_autoencoder = Model(input = input_img, output = encoded3_da)


# In[131]:


autoencoder1.compile(loss='binary_crossentropy', optimizer = 'adam')
autoencoder2.compile(loss='binary_crossentropy', optimizer = 'adam')
autoencoder3.compile(loss='binary_crossentropy', optimizer = 'adam')

encoder1.compile(loss='binary_crossentropy', optimizer = 'adam')
encoder2.compile(loss='binary_crossentropy', optimizer = 'adam')
encoder3.compile(loss='binary_crossentropy', optimizer = 'adam')

deep_autoencoder.compile(loss='binary_crossentropy', optimizer = 'adam')


# In[132]:


history1 = autoencoder1.fit(training_inputs_noisy, training_inputs,
                nb_epoch = 100, batch_size = 512,
                validation_data = (val_inputs_noisy,val_inputs),
                shuffle = True)


# In[133]:


encoder1.layers[1].set_weights(autoencoder1.layers[1].get_weights())
# encoder1.layers[2].set_weights(autoencoder1.layers[2].get_weights())
first_layer_code = encoder1.predict(training_inputs)
first_layer_val_code = encoder1.predict(val_inputs)
print(first_layer_code.shape)


# In[134]:


history2 = autoencoder2.fit(first_layer_code, first_layer_code,
                nb_epoch = 30, batch_size = 512,
                validation_data = (first_layer_val_code,first_layer_val_code),
                shuffle = True)


# In[135]:


encoder2.layers[1].set_weights(autoencoder2.layers[1].get_weights())
# encoder2.layers[2].set_weights(autoencoder2.layers[2].get_weights())
second_layer_code = encoder2.predict(first_layer_code)
second_layer_val_code = encoder2.predict(first_layer_val_code)
print(second_layer_code.shape)


# In[136]:


history3 = autoencoder3.fit(second_layer_code, second_layer_code,
               nb_epoch = 50, batch_size = 512,
               validation_data = (second_layer_val_code,second_layer_val_code),
               shuffle = True)


# In[137]:


deep_autoencoder.layers[1].set_weights(autoencoder1.layers[1].get_weights()) # first dense layer
# deep_autoencoder.layers[2].set_weights(autoencoder1.layers[2].get_weights()) # first bn layer
deep_autoencoder.layers[2].set_weights(autoencoder2.layers[1].get_weights()) # second dense layer
# deep_autoencoder.layers[4].set_weights(autoencoder2.layers[2].get_weights()) # second bn layer
deep_autoencoder.layers[3].set_weights(autoencoder3.layers[1].get_weights()) # thrird dense layer


# In[138]:


dense2 = Dense(5, activation = 'softmax')(encoded3_da)

classifier = Model(input = input_img, output = dense2)

classifier.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])


# In[139]:


history_clf = classifier.fit(training_inputs, training_targets,
                epochs = 90, batch_size = 512,
                validation_data = (val_inputs,val_targets),
                shuffle = True)


# In[146]:


val_preds = classifier.predict(val_inputs)
predictions = np.argmax(val_preds, axis = 1)
true_digits = np.argmax(val_targets, axis = 1)
n_correct = np.sum(np.equal(predictions, true_digits).astype(int))
total = float(len(predictions))
print("Validation Accuracy:", round(n_correct / total, 3))


# In[147]:


training_preds = classifier.predict(training_inputs)
predictions = np.argmax(training_preds, axis = 1)
true_digits = np.argmax(training_targets, axis = 1)
n_correct = np.sum(np.equal(predictions, true_digits).astype(int))
total = float(len(predictions))
print("Training Accuracy:", round(n_correct / total, 3))


# In[140]:


loss_train = history1.history['loss']
loss_val = history1.history['val_loss']
epochs = range(1,101)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[141]:


loss_train = history2.history['loss']
loss_val = history2.history['val_loss']
epochs = range(1,31)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[143]:


loss_train = history3.history['loss']
loss_val = history3.history['val_loss']
epochs = range(1,51)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[145]:


loss_train = history_clf.history['loss']
loss_val = history_clf.history['val_loss']
epochs = range(1,91)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[ ]:




