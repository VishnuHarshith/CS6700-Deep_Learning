# -*- coding: utf-8 -*-
"""Normal.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qFjhscsO29IczwTcIVOW3QjniOHtb04h
"""

import tensorflow as tf
import numpy as np
import unicodedata
import re
import matplotlib.pyplot as plt
import os
import imageio
from zipfile import ZipFile
import string

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def normalize_string_en(s):
    s = unicode_to_ascii(s)
    s = re.sub("'", '', s)
    s = s.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
    s = re.sub(" +", " ", s)
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in s if ch not in exclude)
    s = re.sub(" +", " ", s)
    s = s.strip()
    # s = re.sub(r'([!.?])', r' \1', s)
    # s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    # s = re.sub(r'\s+', r' ', s)
    return s

def normalize_string_hi(s):
    s = re.sub("'", '', s)
    # print(s)
    s = s.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
    # print(s)
    s = re.sub(" +", " ", s)
    # print(s)
    # s = re.sub(r'([!.?])', r' \1', s)
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in s if ch not in exclude)
    # s = re.sub(r'[.!,?]+', r' ', s)
    s = re.sub(" +", " ", s)
    s = s.strip()

    # print(s)
    # s = re.sub(r'\s+', r' ', s)
    return s

MODE = 'train'
BATCH_SIZE = 64
EMBEDDING_SIZE = 100
RNN_SIZE = 512
NUM_EPOCHS = 50
ATTENTION_FUNC = 'concat'

with open('/content/drive/My Drive/Colab Notebooks/train.hi',encoding='utf-8') as f:
    lines_hi = f.readlines()
lines_hi = [x.strip() for x in lines_hi]
with open('/content/drive/My Drive/Colab Notebooks/train.en',encoding='utf-8') as f:
    lines_en = f.readlines()
lines_en = [x.strip() for x in lines_en]

lines_en, lines_hi = list(lines_en), list(lines_hi)
lines_en = [normalize_string_en(data) for data in lines_en]
lines_hi_in = ['<start> ' + normalize_string_hi(data) for data in lines_hi]
lines_hi_out = [normalize_string_hi(data) + ' <end>' for data in lines_hi]

en_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
en_tokenizer.fit_on_texts(lines_en)
lines_en_seq = en_tokenizer.texts_to_sequences(lines_en)
lines_en_seq = tf.keras.preprocessing.sequence.pad_sequences(lines_en_seq,padding='post')

hi_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')

# ATTENTION: always finish with fit_on_texts before moving on
hi_tokenizer.fit_on_texts(lines_hi_in)
hi_tokenizer.fit_on_texts(lines_hi_out)

lines_hi_in_seq = hi_tokenizer.texts_to_sequences(lines_hi_in)
lines_hi_in_seq = tf.keras.preprocessing.sequence.pad_sequences(lines_hi_in_seq,padding='post')

lines_hi_out_seq = hi_tokenizer.texts_to_sequences(lines_hi_out)
lines_hi_out_seq = tf.keras.preprocessing.sequence.pad_sequences(lines_hi_out_seq,padding='post')

dataset = tf.data.Dataset.from_tensor_slices((lines_en_seq, lines_hi_in_seq, lines_hi_out_seq))
dataset = dataset.shuffle(len(lines_en)).batch(BATCH_SIZE, drop_remainder=True)

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

from numpy import array
from numpy import asarray
from numpy import zeros

en_vocab_size = len(en_tokenizer.word_index) + 1
hi_vocab_size = len(hi_tokenizer.word_index) + 1

EMBEDDING_SIZE = 100
LSTM_SIZE = 64



embeddings_dictionary = dict()

glove_file = open(r'/content/drive/My Drive/Colab Notebooks/glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

# num_words = len(word2idx_inputs) + 1
embedding_matrix = zeros((en_vocab_size, 100))
for word, index in en_tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, lstm_size):
        super(Encoder, self).__init__()
        self.lstm_size = lstm_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size, weights=[embedding_matrix],trainable=False, mask_zero = True)
        self.lstm = tf.keras.layers.LSTM(lstm_size, return_sequences=True, return_state=True)

    def call(self, sequence, states):
        embed = self.embedding(sequence)
        output, state_h, state_c = self.lstm(embed, initial_state=states)

        return output, state_h, state_c

    def init_states(self, batch_size):
        return (tf.zeros([batch_size, self.lstm_size]),
                tf.zeros([batch_size, self.lstm_size]))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, lstm_size):
        super(Decoder, self).__init__()
        self.lstm_size = lstm_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.lstm = tf.keras.layers.LSTM(
            lstm_size, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, sequence, state):
        embed = self.embedding(sequence)
        lstm_out, state_h, state_c = self.lstm(embed, state)
        logits = self.dense(lstm_out)

        return logits, state_h, state_c

    def init_states(self, batch_size):
        return (tf.zeros([batch_size, self.lstm_size]),
                tf.zeros([batch_size, self.lstm_size]))




def loss_func(targets, logits):
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, dtype=tf.int64)
    loss = crossentropy(targets, logits, sample_weight=mask)

    return loss


optimizer = tf.keras.optimizers.Adam()


def predict(test_source_text=None):
    if test_source_text is None:
        test_source_text = lines_en[np.random.choice(len(lines_en))]
    print(test_source_text)
    test_source_seq = en_tokenizer.texts_to_sequences([test_source_text])
    print(test_source_seq)

    en_initial_states = encoder.init_states(1)
    en_outputs = encoder(tf.constant(test_source_seq), en_initial_states)

    de_input = tf.constant([[hi_tokenizer.word_index['<start>']]])
    de_state_h, de_state_c = en_outputs[1:]
    out_words = []

    while True:
        de_output, de_state_h, de_state_c = decoder(
            de_input, (de_state_h, de_state_c))
        de_input = tf.argmax(de_output, -1)
        out_words.append(hi_tokenizer.index_word[de_input.numpy()[0][0]])

        if out_words[-1] == '<end>' or len(out_words) >= 20:
            break

    print(' '.join(out_words))
    return out_words

@tf.function
def train_step(source_seq, target_seq_in, target_seq_out, en_initial_states):
    loss = 0
    with tf.GradientTape() as tape:
        en_outputs = encoder(source_seq, en_initial_states)
        en_states = en_outputs[1:]
        de_states = en_states

        de_outputs = decoder(target_seq_in, de_states)
        logits = de_outputs[0]
        loss = loss_func(target_seq_out, logits)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss

encoder = Encoder(en_vocab_size, EMBEDDING_SIZE, LSTM_SIZE)
decoder = Decoder(hi_vocab_size, EMBEDDING_SIZE, LSTM_SIZE)

initial_states = encoder.init_states(1)
encoder_outputs = encoder(tf.constant([[1, 2, 3]]), initial_states)
decoder_outputs = decoder(tf.constant([[1, 2, 3]]), encoder_outputs[1:])

NUM_EPOCHS = 10

for e in range(NUM_EPOCHS):
    en_initial_states = encoder.init_states(BATCH_SIZE)
    
    predict()

    for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
        loss = train_step(source_seq, target_seq_in,target_seq_out, en_initial_states)

    print('Epoch {} Loss {:.4f}'.format(e + 1, loss.numpy()))

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
s = SmoothingFunction().method4

ref = []
can = []
s1,s2,s3,s4 = 0,0,0,0
for i in range(len(lines_en)):
    temp = predict(lines_en[i])
    temp1 = lines_hi[i].split()
    pred = [e for e in temp if e not in ('<start>', '<end>')]
#     print(len(pred),len(temp1))
    s1 += sentence_bleu([pred], temp1, weights=(1, 0, 0, 0), smoothing_function=s)
    s2 += sentence_bleu([pred], temp1, weights=(0.5, 0.5, 0, 0), smoothing_function=s)
    s3 += sentence_bleu([pred], temp1, weights=(0.33, 0.33, 0.33, 0), smoothing_function=s)
    s4 += sentence_bleu([pred], temp1, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=s)