#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt


# Binary RBM for the first data set

# In[2]:


cd /kaggle/input/assin-2-data-2


# In[3]:


X_data, y_data = np.array([]), []
for i,c in enumerate(["T_shirt.csv", "Coat.csv", "Sandal.csv", "Sneaker.csv", "Ankle boot.csv"]):
    curr_data = np.genfromtxt(c, delimiter = ",")
    if not X_data.shape[0]:
        X_data = curr_data
    else:
        X_data = np.vstack([X_data, curr_data])
    y_data.extend([i]*len(curr_data))
y_data = np.array(y_data)


# In[4]:


X_data = X_data / 255.0
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2)


# In[5]:


class bernoulli_rbm():
    def __init__(self, n_hidden, X_train, X_val):
        # define the number of visible units, shape of X is (no of data points, dim of data points)
        n_visible = X_train.shape[1]
        # define the number of hidden units
        n_hidden = n_hidden
        # initialise the weight and the biases
        self.W = tf.random.normal([n_visible, n_hidden])*0.01
        # b is the bias for computing visible units
        self.b_visible = tf.random.normal([n_visible,1])*0.01
        # c is the bias for computing the hidden units
        self.c_hidden = tf.random.normal([n_hidden,1])*0.01
        # the data for which we would fit the RBM for
        self.X_train = tf.constant(X_train, dtype = tf.float32)
        # compute the validation loss
        self.X_val = tf.constant(X_val, dtype = tf.float32)
        # reconstruction loss
        self.reconstruction_loss_train = []
        self.reconstruction_loss_val = []
        # free energy, to detect potential overfitting refer to Prof Hinton's notes
        self.free_energy_train = []
        self.free_energy_val = []

    def k_step_cd(self, train_batch, k):
        # takes a training batch as input and returns the average gradient by running k-cd algorithm
        # return is a tuple of the form (grad_W, grad_b, grad_c)
        batch_size = len(train_batch)
        # hidden values should be sampled for the first time from data
        # after that while computing from reconstructions we can use sample probabilities
        train_batch = tf.transpose(train_batch)
        hidden_prob = tf.nn.sigmoid(tf.matmul(tf.transpose(self.W), train_batch) + self.c_hidden)
        hidden_reconstructed = tf.nn.relu(tf.sign(hidden_prob - tf.random.uniform(hidden_prob.shape)))
        visible_reconstructed = None
        for i in range(k):
            visible_reconstructed = tf.nn.sigmoid(tf.matmul(self.W, hidden_reconstructed) + self.b_visible)
            hidden_reconstructed = tf.nn.sigmoid(tf.matmul(tf.transpose(self.W), visible_reconstructed) + self.c_hidden)
        # now we have completed the gibbs sampling, we will compute the gradients
        # grad of W is vihj_data - vihj_expected
        grad_W = tf.matmul(train_batch, tf.transpose(hidden_prob)) - tf.matmul(visible_reconstructed, tf.transpose(hidden_reconstructed)) / batch_size
        grad_b_visible = tf.reshape(tf.reduce_sum(train_batch - visible_reconstructed, 1),(-1,1)) / batch_size
        grad_c_hidden = tf.reshape(tf.reduce_sum(hidden_prob - hidden_reconstructed, 1),(-1,1)) / batch_size
        return (grad_W, grad_b_visible, grad_c_hidden)

    def run(self, num_epochs = 100, k = 2, lr = 1e-3, batch_size = 10):
        # k is number of times we would run parallel gibbs sampling
        # we run the stochastic gradient ascent algorithm for num_epochs times
        # with the given batch size
        training_rows = len(self.X_train)
        for i in range(num_epochs):
            # compute the loss and record it
            self.loss_compute()
            print("epoch",i+1," loss:",self.reconstruction_loss_train[-1], self.reconstruction_loss_val[-1], self.free_energy_train[-1], self.free_energy_val[-1])
            for j in range(training_rows // batch_size + 1):
                curr_batch = self.X_train[j*batch_size : (j+1)*batch_size]
                if not curr_batch.shape[0]:
                    continue
                # get the gradients by running k-CD algorithm, remember these are the "mean" gradients
                grad_W, grad_b_visible, grad_c_hidden = self.k_step_cd(curr_batch, k)
                # update the parameters
                self.W = self.W + lr*grad_W
                self.b_visible = self.b_visible + lr*grad_b_visible
                self.c_hidden = self.c_hidden + lr*grad_c_hidden
    
    def loss_compute(self):
        # compute the training and validation reconstruction loss
        hidden_representation = tf.nn.sigmoid(tf.matmul(tf.transpose(self.W), tf.transpose(self.X_train)) + self.c_hidden)
        reconstruction = tf.nn.sigmoid(tf.matmul(self.W, hidden_representation) + self.b_visible)
        reconstruction_error = tf.reduce_sum((tf.transpose(self.X_train) - reconstruction) ** 2) / len(self.X_train)
        self.reconstruction_loss_train.append(float(reconstruction_error))
        hidden_representation = tf.nn.sigmoid(tf.matmul(tf.transpose(self.W), tf.transpose(self.X_val)) + self.c_hidden)
        reconstruction = tf.nn.sigmoid(tf.matmul(self.W, hidden_representation) + self.b_visible)
        reconstruction_error = tf.reduce_sum((tf.transpose(self.X_val) - reconstruction) ** 2) / len(self.X_val)
        self.reconstruction_loss_val.append(float(reconstruction_error))
        # compute the training and the validation average free energy
        train_free_energy = (-tf.reduce_sum(tf.multiply(tf.transpose(self.X_train), self.b_visible))-tf.reduce_sum(tf.math.log(1+tf.exp(tf.matmul(tf.transpose(self.W),tf.transpose(self.X_train))+self.c_hidden)))) / len(self.X_train)
        val_free_energy = (-tf.reduce_sum(tf.multiply(tf.transpose(self.X_val), self.b_visible))-tf.reduce_sum(tf.math.log(1+tf.exp(tf.matmul(tf.transpose(self.W),tf.transpose(self.X_val))+self.c_hidden)))) / len(self.X_val)
        self.free_energy_train.append(float(train_free_energy))
        self.free_energy_val.append(float(val_free_energy))

    def plotter(self):
        # function to plot the various cost measures over time
        plt.figure(figsize = (12,7))
        plt.plot(range(1,len(self.free_energy_train)+1), self.free_energy_train, c = "r", label = "free energy train")
        plt.plot(range(1,len(self.free_energy_val)+1), self.free_energy_val, c = "g", label = "free energy val")
        plt.xlabel("Epochs")
        plt.ylabel("free energy")
        plt.legend()
        plt.show()
        plt.figure(figsize = (12,7))
        plt.plot(range(1,len(self.reconstruction_loss_train)+1), self.reconstruction_loss_train, c = "r", label = "reconstruction error train")
        plt.plot(range(1,len(self.reconstruction_loss_val)+1), self.reconstruction_loss_val, c = "g", label = "reconstruction error val")
        plt.xlabel("Epochs")
        plt.ylabel("reconstruction error")
        plt.legend()
        plt.show()


# In[6]:


# train the first layer of RBM
a = bernoulli_rbm(n_hidden=200, X_train=X_train, X_val=X_test)
a.run(num_epochs=50, k = 1, lr = 1e-3, batch_size = 1)
a.plotter()


# In[7]:


# first reduce the dimension of the input
X_train_1 =  tf.transpose(tf.nn.sigmoid(tf.matmul(tf.transpose(a.W), tf.transpose(tf.constant(X_train,dtype=tf.float32))) + a.c_hidden))
X_val_1 =  tf.transpose(tf.nn.sigmoid(tf.matmul(tf.transpose(a.W), tf.transpose(tf.constant(X_test,dtype=tf.float32))) + a.c_hidden))
# now train the second layer of RBM
b = bernoulli_rbm(n_hidden=100, X_train=X_train_1, X_val=X_val_1)
b.run(num_epochs=50, k = 1, lr = 1e-3, batch_size = 1)
b.plotter()


# In[8]:


# first reduce the dimension of the input
X_train_2 =  tf.transpose(tf.nn.sigmoid(tf.matmul(tf.transpose(b.W), tf.transpose(X_train_1)) + b.c_hidden))
X_val_2 =  tf.transpose(tf.nn.sigmoid(tf.matmul(tf.transpose(b.W), tf.transpose(X_val_1)) + b.c_hidden))
# now train the second layer of RBM
c = bernoulli_rbm(n_hidden=50, X_train=X_train_2, X_val=X_val_2)
c.run(num_epochs=50, k = 1, lr = 1e-3, batch_size = 1)
c.plotter()


# In[9]:


X_data.shape


# In[10]:


# define a neural net with the same layers as above
inputs = tf.keras.Input(shape = (784,))
x = tf.keras.layers.Dense(200, activation = "relu")(inputs)
x = tf.keras.layers.Dense(100, activation = "relu")(x)
x = tf.keras.layers.Dense(50, activation = "relu")(x)
outputs = tf.keras.layers.Dense(5)(x)
model = tf.keras.Model(inputs = inputs, outputs = outputs, name = "binaryrbmmodel")
# set weights of the keras model with the rbm ones
model.layers[1].set_weights([a.W, tf.squeeze(a.c_hidden)])
model.layers[2].set_weights([b.W, tf.squeeze(b.c_hidden)])
model.layers[3].set_weights([c.W, tf.squeeze(c.c_hidden)])
# compile the model
model.compile(optimizer= tf.keras.optimizers.Adam(lr = 1e-3),
              loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)
model.summary()


# In[11]:


history = model.fit(X_train, y_train, epochs = 500, batch_size = 256, validation_data = (X_test, y_test), callbacks = [callback], verbose = 0)


# In[12]:


plt.figure(figsize = (12,7))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss data set 2')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')


# In[13]:


plt.figure(figsize = (12,7))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy data set 2')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')


# Gaussian RBM for the second data set:

# In[14]:


## NOTE: data is expected to be normalised to mean 0 and var 1
class gaussian_rbm():
    def __init__(self, n_hidden, X_train, X_val):
        # define the number of visible units, shape of X is (no of data points, dim of data points)
        n_visible = X_train.shape[1]
        # define the number of hidden units
        n_hidden = n_hidden
        # initialise the weight and the biases
        self.W = tf.random.normal([n_visible, n_hidden])*0.01
        # b is the bias for computing visible units
        self.b_visible = tf.random.normal([n_visible,1])*0.01
        # c is the bias for computing the hidden units
        self.c_hidden = tf.random.normal([n_hidden,1])*0.01
        # the data for which we would fit the RBM for
        self.X_train = tf.constant(X_train, dtype = tf.float32)
        # compute the validation loss
        self.X_val = tf.constant(X_val, dtype = tf.float32)
        # reconstruction loss
        self.reconstruction_loss_train = []
        self.reconstruction_loss_val = []
        # free energy, to detect potential overfitting refer to Prof Hinton's notes
        self.free_energy_train = []
        self.free_energy_val = []

    def k_step_cd(self, train_batch, k):
        # takes a training batch as input and returns the average gradient by running k-cd algorithm
        # return is a tuple of the form (grad_W, grad_b, grad_c)
        batch_size = len(train_batch)
        # hidden values should be sampled for the first time from data
        # after that while computing from reconstructions we can use sample probabilities
        train_batch = tf.transpose(train_batch)
        hidden_prob = tf.nn.sigmoid(tf.matmul(tf.transpose(self.W), train_batch) + self.c_hidden)
        ## this changes for gaussian RBM
        hidden_reconstructed = tf.nn.relu(tf.sign(hidden_prob - tf.random.uniform(hidden_prob.shape)))
        visible_reconstructed = None
        for i in range(k):
            visible_reconstructed = tf.matmul(self.W, hidden_reconstructed) + self.b_visible
            visible_reconstructed = tf.random.normal(visible_reconstructed.shape) + visible_reconstructed
            hidden_reconstructed = tf.nn.sigmoid(tf.matmul(tf.transpose(self.W), visible_reconstructed) + self.c_hidden)
        # now we have completed the gibbs sampling, we will compute the gradients
        # grad of W is vihj_data - vihj_expected
        grad_W = tf.matmul(train_batch, tf.transpose(hidden_prob)) - tf.matmul(visible_reconstructed, tf.transpose(hidden_reconstructed)) / batch_size
        grad_b_visible = tf.reshape(tf.reduce_sum(train_batch - visible_reconstructed, 1),(-1,1)) / batch_size
        grad_c_hidden = tf.reshape(tf.reduce_sum(hidden_prob - hidden_reconstructed, 1),(-1,1)) / batch_size
        return (grad_W, grad_b_visible, grad_c_hidden)

    def run(self, num_epochs = 100, k = 2, lr = 1e-3, batch_size = 10):
        # k is number of times we would run parallel gibbs sampling
        # we run the stochastic gradient ascent algorithm for num_epochs times
        # with the given batch size
        training_rows = len(self.X_train)
        for i in range(num_epochs):
            if not (i+1) % 15:
                print("here")
                lr = lr / 2
            self.loss_compute()
            print("epoch",i+1," loss:",self.reconstruction_loss_train[-1], self.reconstruction_loss_val[-1], self.free_energy_train[-1], self.free_energy_val[-1])
            for j in range(training_rows // batch_size + 1):
                curr_batch = self.X_train[j*batch_size : (j+1)*batch_size]
                if not curr_batch.shape[0]:
                    continue
                # get the gradients by running k-CD algorithm, remember these are the "mean" gradients
                grad_W, grad_b_visible, grad_c_hidden = self.k_step_cd(curr_batch, k)
                # before updating record the loss
                # update the parameters
                self.W = self.W + lr*grad_W
#                 print(tf.reduce_sum(self.W))
                self.b_visible = self.b_visible + lr*grad_b_visible
                self.c_hidden = self.c_hidden + lr*grad_c_hidden
    
    def loss_compute(self):
        # compute the training and validation reconstruction loss
        hidden_representation = tf.nn.sigmoid(tf.matmul(tf.transpose(self.W), tf.transpose(self.X_train)) + self.c_hidden)
        reconstruction = tf.matmul(self.W, hidden_representation) + self.b_visible
        reconstruction = reconstruction + tf.random.normal(reconstruction.shape)
        reconstruction_error = tf.reduce_sum((tf.transpose(self.X_train) - reconstruction) ** 2) / len(self.X_train)
        self.reconstruction_loss_train.append(float(reconstruction_error))
        hidden_representation = tf.nn.sigmoid(tf.matmul(tf.transpose(self.W), tf.transpose(self.X_val)) + self.c_hidden)
        reconstruction = tf.matmul(self.W, hidden_representation) + self.b_visible
        reconstruction = reconstruction + tf.random.normal(reconstruction.shape)
        reconstruction_error = tf.reduce_sum((tf.transpose(self.X_val) - reconstruction) ** 2) / len(self.X_val)
        self.reconstruction_loss_val.append(float(reconstruction_error))
        # compute the training and the validation average free energy
        train_free_energy = (-tf.reduce_sum(tf.multiply(tf.transpose(self.X_train), self.b_visible))-tf.reduce_sum(tf.math.log(1+tf.exp(tf.matmul(tf.transpose(self.W),tf.transpose(self.X_train))+self.c_hidden)))) / len(self.X_train)
        val_free_energy = (-tf.reduce_sum(tf.multiply(tf.transpose(self.X_val), self.b_visible))-tf.reduce_sum(tf.math.log(1+tf.exp(tf.matmul(tf.transpose(self.W),tf.transpose(self.X_val))+self.c_hidden)))) / len(self.X_val)
        self.free_energy_train.append(float(train_free_energy))
        self.free_energy_val.append(float(val_free_energy))

    def plotter(self):
        # function to plot the various cost measures over time
        plt.figure(figsize = (12,7))
        plt.plot(range(1,len(self.free_energy_train)+1), self.free_energy_train, c = "r", label = "free energy train")
        plt.plot(range(1,len(self.free_energy_val)+1), self.free_energy_val, c = "g", label = "free energy val")
        plt.xlabel("Epochs")
        plt.ylabel("free energy")
        plt.legend()
        plt.show()
        plt.figure(figsize = (12,7))
        plt.plot(range(1,len(self.reconstruction_loss_train)+1), self.reconstruction_loss_train, c = "r", label = "reconstruction error train")
        plt.plot(range(1,len(self.reconstruction_loss_val)+1), self.reconstruction_loss_val, c = "g", label = "reconstruction error val")
        plt.xlabel("Epochs")
        plt.ylabel("reconstruction error")
        plt.legend()
        plt.show()


# In[15]:


cd /kaggle/input/assin2-data-1/"Data_Set_1(Colored_Images)"


# In[16]:


X_data = []
y_data = []
# get the number of images in each class
for i in ["insidecity", "street", "highway", "mountain", "opencountry"]:
  print(i, len(os.listdir(i+"/"+i)))
# get the number of images in each class
class_val = 0
for i in ["insidecity", "street", "highway", "mountain", "opencountry"]:
  for j in os.listdir(i+"/"+i):
    x = np.loadtxt(i+"/"+i+"/"+j)
    X_data.append(x)
    y_data.append(class_val)
  class_val += 1
X_data = np.array(X_data)
y_data = np.array(y_data)
X_data = X_data.reshape(1644,-1)
print("shape of X:",X_data.shape)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2)


# In[17]:


# train gaussian RBM
a = gaussian_rbm(n_hidden=200, X_train=X_train, X_val=X_test)
a.run(num_epochs=50, k = 1, lr = 1e-6, batch_size = 1)
a.plotter()


# In[18]:


# first reduce the dimension of the input
X_train_1 =  tf.transpose(tf.nn.sigmoid(tf.matmul(tf.transpose(a.W), tf.transpose(tf.constant(X_train,dtype=tf.float32))) + a.c_hidden))
X_val_1 =  tf.transpose(tf.nn.sigmoid(tf.matmul(tf.transpose(a.W), tf.transpose(tf.constant(X_test,dtype=tf.float32))) + a.c_hidden))
# now train the second layer of RBM
b = gaussian_rbm(n_hidden=100, X_train=X_train_1, X_val=X_val_1)
b.run(num_epochs=31, k = 1, lr = 1e-5, batch_size = 1)
b.plotter()


# In[19]:


# first reduce the dimension of the input
X_train_2 =  tf.transpose(tf.nn.sigmoid(tf.matmul(tf.transpose(b.W), tf.transpose(X_train_1)) + b.c_hidden))
X_val_2 =  tf.transpose(tf.nn.sigmoid(tf.matmul(tf.transpose(b.W), tf.transpose(X_val_1)) + b.c_hidden))
# now train the second layer of RBM
c = gaussian_rbm(n_hidden=50, X_train=X_train_2, X_val=X_val_2)
c.run(num_epochs=31, k = 1, lr = 1e-5, batch_size = 1)
c.plotter()


# In[20]:


# define a neural net with the same layers as above
inputs = tf.keras.Input(shape = (828,))
x = tf.keras.layers.Dense(200, activation = "sigmoid")(inputs) # <-- layer 1
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(100, activation = "sigmoid")(x)      # <-- layer 4
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(50, activation = "sigmoid")(x)       # <-- layer 7
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(5)(x)
model = tf.keras.Model(inputs = inputs, outputs = outputs, name = "gaussianrbmmodel")
# set weights of the keras model with the rbm ones
model.layers[1].set_weights([a.W, tf.squeeze(a.c_hidden)])
model.layers[4].set_weights([b.W, tf.squeeze(b.c_hidden)])
model.layers[7].set_weights([c.W, tf.squeeze(c.c_hidden)])


# In[21]:


# compile the model
model.compile(optimizer= tf.keras.optimizers.Adam(lr = 5e-5),
              loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience = 20)
model.summary()


# In[22]:


history = model.fit(X_train, y_train, epochs = 1000, batch_size = 256, validation_data = (X_test, y_test), verbose = 0)


# In[23]:


plt.figure(figsize = (12,7))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss data set 2')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')


# In[24]:


# plot the train and val accuracies
plt.figure(figsize = (12,7))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy data set 2')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
print(history.history['val_accuracy'][-1])

