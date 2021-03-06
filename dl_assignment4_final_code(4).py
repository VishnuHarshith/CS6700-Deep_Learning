# -*- coding: utf-8 -*-
"""DL_Assignment4_Final_code.ipynb

Automatically generated by Colaboratory.


"""

from google.colab import drive

drive.mount('/content/gdrive')

# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as ppvgg
from tensorflow.keras.applications.inception_v3 import preprocess_input as ppggl
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras import datasets, layers, models

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences as ps
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.regularizers import l2

import os
import pandas as pd
import numpy as np
import difflib
import glob
import string
import cv2
import matplotlib.pyplot as plt



import pickle 
from pickle import dump, load

from time import time

# %matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

"""### Extract Images"""

import zipfile

import zipfile

with zipfile.ZipFile(
    "/content/gdrive/My Drive/DL_Assignment4/images.zip", "r"
) as zip_ref:
    zip_ref.extractall("/content/gdrive/My Drive/DL_Assignment4/images")

with zipfile.ZipFile(
    "/content/gdrive/My Drive/DL_Assignment4/glove.6B.zip", "r"
) as zip_ref:
    zip_ref.extractall("/content/gdrive/My Drive/DL_Assignment4/Glove/")

"""### Loading Captions"""

# load doc into memory
def read_files(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

filename = "/content/gdrive/My Drive/DL_Assignment4/captions.txt"
# load captions
images_with_captions = read_files(filename)

def make_dictionary(doc):
	mapping = dict()
	# process lines
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		if len(line) < 2:
			continue
		# take the first token as the image id, the rest as the caption
		image_id, image_caption = tokens[0], tokens[1:]
		# extract filename from image id
		image_id = image_id.split('.')[0]
		# convert caption tokens back to string
		image_caption = ' '.join(image_caption)
		# create the list if needed
		if image_id not in mapping:
			mapping[image_id] = list()
		# store caption
		mapping[image_id].append(image_caption)
	return mapping

# parse captions
images_with_captions = make_dictionary(images_with_captions)

"""### Cleaning Captions - No need to run it twice"""

def clean_captions(captions):
	# for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for key, desc_list in captions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]
			# tokenize
			desc = desc.split()
			# lower case
			desc = [word.lower() for word in desc]
			# remove punctuation 
			desc = [w.translate(table) for w in desc]
			# remove hanging letters
			desc = [word for word in desc if len(word)>1]
			# remove alphanumeric letters
			desc = [word for word in desc if word.isalpha()]
			# store as string
			desc_list[i] =  ' '.join(desc)

# clean captions
clean_captions(images_with_captions)

#make vocabulary
def make_vocabulary(captions):
	# build a list of all description strings
	all_captions = set()
	for key in captions.keys():
		[all_captions.update(d.split()) for d in captions[key]]
	return all_captions

vocabulary = make_vocabulary(images_with_captions)
print('Vocabulary Size: %d' % len(vocabulary))

#save files
def save_captions(captions, filename):
	lines = list()
	for key, desc_list in captions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

save_captions(images_with_captions, '/content/gdrive/My Drive/DL_Assignment4/descriptions.txt')

"""### Load Images"""

# load a pre-defined list of photo identifiers
def load_dataset(filename):
	doc = read_files(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

# load training dataset (6K)
filename = '/content/gdrive/My Drive/DL_Assignment4/trainImages.txt'
train = load_dataset(filename)
print('TrainDataset: %d' % len(train))

# All images are here
images = '/content/gdrive/My Drive/DL_Assignment4/images/Images/'
# Extracting them
img = glob.glob(images + '*.jpg')

# Train Data
train_images_file = '/content/gdrive/My Drive/DL_Assignment4/trainImages.txt'
# Read it in a set
train_images = set(open(train_images_file, 'r').read().strip().split('\n'))

# Create a list
train_img = []

for i in img: # contains all images names
    if i[len(images):] in train_images: # Check if the image belongs to training set # Can decrease training Dataset here
        train_img.append(i) # Add it to list

# Below file conatains the names of images to be used in test data
test_images_file = '/content/gdrive/My Drive/DL_Assignment4/testImages.txt'
test = load_dataset(test_images_file)
# Read the validation image names in a set# Read the test image names in a set
test_images = set(open(test_images_file, 'r').read().strip().split('\n'))

# Create a list 
test_img = []

for i in img: # contains all images
    if i[len(images):] in test_images: # Check if the image belongs to test set
        test_img.append(i) # Add it

# Below file conatains the names of images to be used in test data
val_images_file = '/content/gdrive/My Drive/DL_Assignment4/valImages.txt'
# Read the validation image names in a set# Read the test image names in a set
val = load_dataset(val_images_file)
val_images = set(open(val_images_file, 'r').read().strip().split('\n'))

# Create a list 
val_img = []

for i in img: # contains all images
    if i[len(images):] in val_images: # Check if the image belongs to test set
        val_img.append(i) # Add it

images_with_captions= read_files('/content/gdrive/My Drive/DL_Assignment4/descriptions.txt')

# load clean descriptions
def make_clean_descriptions(filename, dataset):
	# load document
	doc = read_files(filename)
	captions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in captions:
				captions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			captions[image_id].append(desc)
	return captions

# dividing captions
train_captions = make_clean_descriptions('/content/gdrive/My Drive/DL_Assignment4/descriptions.txt', train)
val_captions = make_clean_descriptions('/content/gdrive/My Drive/DL_Assignment4/descriptions.txt', val)
test_captions = make_clean_descriptions('/content/gdrive/My Drive/DL_Assignment4/descriptions.txt', test)

"""### No need to run this twice -- Creating Input datasets"""

SHAPE_WIDTH = 224
SHAPE_HEIGHT = 224

# resize pack to fixed size SHAPE_WIDTH x SHAPE_HEIGHT
def resize_pack(pack):
    fx_ratio = SHAPE_WIDTH / pack.shape[1]
    fy_ratio = SHAPE_HEIGHT / pack.shape[0]    
    pack = cv2.resize(pack, (0, 0), fx=fx_ratio, fy=fy_ratio)
    return pack[0:SHAPE_HEIGHT, 0:SHAPE_WIDTH]

val_features = {}
for img in val:
  arr = cv2.imread('/content/gdrive/My Drive/DL_Assignment4/images/Images/'+img+'.jpg')
  vgg_arr = resize_pack(arr)
  val_features[img] = vgg_arr

test_features = {}
for img in test:
  arr = cv2.imread('/content/gdrive/My Drive/DL_Assignment4/images/Images/'+img+'.jpg')
  vgg_arr = resize_pack(arr)
  test_features[img] = vgg_arr

train_features = {}
for img in train:
  arr = cv2.imread('/content/gdrive/My Drive/DL_Assignment4/images/Images/'+img+'.jpg')
  vgg_arr = resize_pack(arr)
  train_features[img] = vgg_arr

with open('/content/gdrive/My Drive/DL_Assignment4/vggtrain_features.pkl', "wb") as encoded_pickle:
    pickle.dump(train_features, encoded_pickle)
with open('/content/gdrive/My Drive/DL_Assignment4/vggval_features.pkl', "wb") as encoded_pickle:
    pickle.dump(val_features, encoded_pickle)
with open('/content/gdrive/My Drive/DL_Assignment4/vggtest_features.pkl', "wb") as encoded_pickle:
    pickle.dump(test_features, encoded_pickle)

"""### Run from here"""

test_features = pickle.load(open('/content/gdrive/My Drive/DL_Assignment4/vggtest_features.pkl', 'rb'))
val_features = pickle.load(open('/content/gdrive/My Drive/DL_Assignment4/vggval_features.pkl', 'rb'))
train_features = pickle.load(open('/content/gdrive/My Drive/DL_Assignment4/vggtrain_features.pkl', 'rb'))

# Create a list of all the training captions
all_train_captions = []
for key, value in train_captions.items():
    for cap in value:
        all_train_captions.append(cap)
len(all_train_captions)

"""### Using pre-trained Glove Reprsentation"""

# Consider only words which occur at least once in the corpus
word_count_threshold = 1
word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
print('preprocessed words %d -> %d' % (len(word_counts), len(vocab)))

ixtoword = {}
wordtoix = {}

ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

vocab_size = len(ixtoword) + 1 # one for appended 0's

# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(captions):
	all_captions = list()
	for key in captions.keys():
		[all_captions.append(d) for d in captions[key]]
	return all_captions

# calculate the length of the description with the most words
def max_length(captions):
	lines = to_lines(captions)
	return max(len(d.split()) for d in lines)

# determine the maximum sequence length
max_length = max_length(train_captions)
print('Caption Length: %d' % max_length)

# Load Glove vectors
glove_dir = '/content/gdrive/My Drive/DL_Assignment4/Glove/'
embeddings_index = {} # empty dictionary
f = open(os.path.join(glove_dir, 'glove.6B.200d.txt'), encoding="utf-8")

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

vocab_size

embedding_dim = 200

# Get 200-dim dense vector for each of the 7579 words
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in wordtoix.items():
    #if i < max_words:
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in the embedding index will be all zeros
        embedding_matrix[i] = embedding_vector

"""### Data Generator"""

def data_generator(captions, photos, wordtoix, max_length, num_photos_per_batch):
    global vocab_size
    X1, X2, y = list(), list(), list()
    n=0
    # loop for ever over images
    while 1:
        for key, desc_list in captions.items():
            n+=1
            # retrieve the photo feature
            photo = photos[key]
            for desc in desc_list:
                # encode the sequence
                seq = [wordtoix[word] for word in desc.split() if word in wordtoix]
                in_seq = seq[:-1]
                in_seq = ps([in_seq], maxlen=max_length, padding='post')[0]
                # encode output sequence
                out_seq = seq[1:]
                out_seq = ps([out_seq], maxlen=max_length, padding='post')[0]
                X1.append(photo)
                X2.append(np.array(in_seq))
                out_seq = to_categorical(out_seq, num_classes=vocab_size)
                y.append(np.array(out_seq))
            # yield the batch data
            if n==num_photos_per_batch:
                yield [np.array(X1), np.array(X2)], np.array(y)
                X1, X2, y = list(), list(), list()
                n=0

"""### Creating Model"""

from keras import backend as K
from keras import layers, models 
class NetVLAD(layers.Layer):
    def __init__(self, num_clusters, cluster_initializer=None, skip_postnorm=False, **kwargs):
        self.K = num_clusters
        self.skip_postnorm = skip_postnorm
        super(NetVLAD, self).__init__(**kwargs)

    def build(self, input_shape):
        # self.K = 
        self.D = input_shape[-1]
        self.C = self.add_weight(name='cluster_centers',
                                    shape=(1,1,1,self.D,self.K),
                                    initializer='random_normal',
                                    dtype='float32',
                                    trainable=True)
        self.kernel = self.add_weight(name = 'kernel',
                                        shape = (1, 1, self.D, self.K),
                                        initializer = 'glorot_uniform',
                                        dtype = 'float32',
                                        trainable = True)
        self.bias = self.add_weight(name = 'bias',
                                        shape = (1,1,self.K),
                                        initializer = 'random_normal',
                                        dtype = 'float32',
                                        trainable = True)

    
        super(NetVLAD, self).build(input_shape) # Be sure to call this at the end
    
    def get_config(self):
        # For serialization with 'custom_objects'
        config = super().get_config()
        config['num_clusters'] = self.K
        return config

    def call(self, inputs):
        s = K.conv2d(inputs, self.kernel, padding = 'same') + self.bias
        a = K.softmax(s)

        # Dims used hereafter: batch, H, W, desc_coeff, cluster
        # Move cluster assignment to corresponding dimension.
        a = tf.expand_dims(a,-2)

        # VLAD core.
        v = tf.expand_dims(inputs,-1)+self.C
        v = a*v
        v = tf.reduce_sum(v,axis=[1,2])
        v = tf.transpose(v,perm=[0,2,1])
        # v = layers.Flatten()(v)

        if not self.skip_postnorm:
            # Result seems to be very sensitive to the normalization method
            # details, so sticking to matconvnet-style normalization here.
            v = self.matconvnetNormalize(v, 1e-12)
            v = tf.transpose(v, perm=[0, 2, 1])
            v = self.matconvnetNormalize(layers.Flatten()(v), 1e-12)
        return v

    def matconvnetNormalize(self,inputs, epsilon):
        return inputs / tf.sqrt(tf.reduce_sum(inputs ** 2, axis=-1, keepdims=True)
        + epsilon)

    def compute_output_shape(self, input_shape):
        return tuple([None, self.K * self.D])

class SimpleAdd(layers.Layer):
    def __init__(self, input_dim):
        super(SimpleAdd, self).__init__()
        c_init = tf.random_normal_initializer()
        self.c = tf.Variable(initial_value=c_init(shape=input_dim, dtype='float32'),trainable=True)

    def call(self, inputs):
        return inputs + self.c

input_layer1 = layers.Input(shape=(224,224,3), name='input_layer1')
#change this for Inceptionv3
encoder = keras.applications.VGG16(weights="imagenet", include_top=False, input_tensor=input_layer1)
encoder.layers.pop()
encoder = models.Model(encoder.input, encoder.output)
for layer in encoder.layers:
    layer.trainable = False
enc_out = encoder.output
vlad = NetVLAD(num_clusters = 32)(enc_out)
vlad_drop = layers.Dropout(0)(vlad)
enc1= layers.Dense(512, activation='relu')(vlad_drop)
input_layer2 = layers.Input(shape=(None,))
embed = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(input_layer2)
embed_drop = layers.Dropout(0)(embed)
#change this for LSTM
decoder = layers.SimpleRNN(512, name='decoder', return_sequences=True)(embed_drop, initial_state=enc1)
dec_out = layers.Dense(vocab_size, activation='softmax')(decoder) 
final_model = models.Model(inputs = [input_layer1,input_layer2], outputs = dec_out)
final_model.layers[-6].set_weights([embedding_matrix])
final_model.layers[-6].trainable = False
final_model.summary()
print(final_model.layers[-6])

final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

"""### Pre-processing techniques"""

val_features_vgg = {}
for img in val:
  arr = val_features[img].astype('float32')
  arr /= 255
  arr = ppvgg(arr)
  val_features_vgg[img] = arr

train_features_vgg = {}
for img in train:
  arr = train_features[img].astype('float32')
  arr /= 255
  arr = ppvgg(arr)
  train_features_vgg[img] = arr

test_features_vgg = {}
for img in test:
  arr = test_features[img].astype('float32')
  arr /= 255
  arr = ppvgg(arr)
  test_features_vgg[img] = arr

"""### Run the Model (Yay!)"""

len(train_captions)

len(train_features_vgg)

import math
epochs = 20
number_pics_per_batch = 8
steps = len(train_descriptions)//number_pics_per_batch

for i in range(epochs):
    print(i)
    final_model.fit_generator(data_generator(train_captions, train_features_vgg, wordtoix, max_length, number_pics_per_batch), 
                         validation_data = data_generator(val_captions, val_features_vgg, wordtoix, max_length, number_pics_per_batch), 
                         validation_steps = math.ceil(len(val_features_vgg)/number_pics_per_batch),
                         steps_per_epoch=steps, 
                         epochs=1)
     
    
    final_model.save('/content/gdrive/My Drive/DL_Assignment4'+'/vggmodel_' + str(i)+'.h5')

"""### Evaluate Results"""

final_model = keras.models.load_model('/content/gdrive/My Drive/DL_Assignment4'+'/vggmodel_' + str(19)+'.h5',  custom_objects={'NetVLAD':NetVLAD, 'num_clusters':32})

def greedySearch(mod, photo):
    in_text = 'startseq'
    photo = np.array([photo])
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = ps([sequence], maxlen=max_length, padding='post')
        yhat = mod.predict([photo,sequence])[0][i]
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

predicted_test = []
for z in range(100):
  pic = list(test_captions_bleu.keys())[z]
  image = test_features_vgg[pic]
  predicted_test.append([greedySearch(final_model, image)])

"""### Bleu Scores"""

import nltk

# load clean descriptions
def clean_descriptions(filename, dataset):
	# load document
	doc = read_files(filename)
	captions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in captions:
				captions[image_id] = list()
			# wrap description in tokens
			desc = ' '.join(image_desc) 
			# store
			captions[image_id].append(desc)
	return captions

# dividing captions
# train_captions = make_clean_descriptions('/content/gdrive/My Drive/DL_Assignment4/descriptions.txt', train)
# val_captions = make_clean_descriptions('/content/gdrive/My Drive/DL_Assignment4/descriptions.txt', val)
test_captions_bleu = clean_descriptions('/content/gdrive/My Drive/DL_Assignment4/descriptions.txt', test)

l1 = list(test_captions_bleu.keys())

originaltest_captions = []
for i in l1[:100]:
  originaltest_captions.append(test_captions_bleu[i][0].split())

pred = []
for i in range(len(predicted_test)):
  pred.append(predicted_test[i][0].split())

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

for k,weights in enumerate([(1,0,0,0), (0.5,0.5,0,0), (0.33,0.33,0.33,0), (0.25,0.25,0.25,0.25)]):
    train_score = []
    for i in range(len(originaltest_captions)):
        s = [originaltest_captions[i]]
        t = pred[i]
        train_score.append(sentence_bleu(s, t, weights = weights))
      
    print("train BLEU score @",k+1,":", np.mean(train_score))

