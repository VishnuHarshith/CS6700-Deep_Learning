# -*- coding: utf-8 -*-
"""Attn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Xl_bjgpC_bbayUqCRBf85XRX22bU2Jkc
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
    # printa(s)
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
NUM_EPOCHS = 10
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

from numpy import array
from numpy import asarray
from numpy import zeros

en_vocab_size = len(en_tokenizer.word_index) + 1
hi_vocab_size = len(hi_tokenizer.word_index) + 1

EMBEDDING_SIZE = 100
RNN_SIZE = 64


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
    def __init__(self, vocab_size, embedding_size, rnn_size):
        super(Encoder, self).__init__()
        self.rnn_size = rnn_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size, weights=[embedding_matrix],trainable=False, mask_zero = True)
        self.lstm = tf.keras.layers.LSTM(
            rnn_size, return_sequences=True, return_state=True)

    def call(self, sequence, states):
        embed = self.embedding(sequence)
        output, state_h, state_c = self.lstm(embed, initial_state=states)

        return output, state_h, state_c

    def init_states(self, batch_size):
        return (tf.zeros([batch_size, self.rnn_size]),
                tf.zeros([batch_size, self.rnn_size]))


# en_vocab_size = len(en_tokenizer.word_index) + 1

encoder = Encoder(en_vocab_size, EMBEDDING_SIZE, RNN_SIZE)

class LuongAttention(tf.keras.Model):
    def __init__(self, rnn_size, attention_func):
        super(LuongAttention, self).__init__()
        self.attention_func = attention_func

        if attention_func not in ['dot', 'general', 'concat']:
            raise ValueError(
                'Unknown attention score function! Must be either dot, general or concat.')

        if attention_func == 'general':
            # General score function
            self.wa = tf.keras.layers.Dense(rnn_size)
        elif attention_func == 'concat':
            # Concat score function
            self.wa = tf.keras.layers.Dense(rnn_size, activation='tanh')
            self.va = tf.keras.layers.Dense(1)

    def call(self, decoder_output, encoder_output):
        if self.attention_func == 'dot':
            # Dot score function: decoder_output (dot) encoder_output
            # decoder_output has shape: (batch_size, 1, rnn_size)
            # encoder_output has shape: (batch_size, max_len, rnn_size)
            # => score has shape: (batch_size, 1, max_len)
            score = tf.matmul(decoder_output, encoder_output, transpose_b=True)
        elif self.attention_func == 'general':
            # General score function: decoder_output (dot) (Wa (dot) encoder_output)
            # decoder_output has shape: (batch_size, 1, rnn_size)
            # encoder_output has shape: (batch_size, max_len, rnn_size)
            # => score has shape: (batch_size, 1, max_len)
            score = tf.matmul(decoder_output, self.wa(
                encoder_output), transpose_b=True)
        elif self.attention_func == 'concat':
            # Concat score function: va (dot) tanh(Wa (dot) concat(decoder_output + encoder_output))
            # Decoder output must be broadcasted to encoder output's shape first
            decoder_output = tf.tile(
                decoder_output, [1, encoder_output.shape[1], 1])

            # Concat => Wa => va
            # (batch_size, max_len, 2 * rnn_size) => (batch_size, max_len, rnn_size) => (batch_size, max_len, 1)
            score = self.va(
                self.wa(tf.concat((decoder_output, encoder_output), axis=-1)))

            # Transpose score vector to have the same shape as other two above
            # (batch_size, max_len, 1) => (batch_size, 1, max_len)
            score = tf.transpose(score, [0, 2, 1])

        # alignment a_t = softmax(score)
        alignment = tf.nn.softmax(score, axis=2)

        # context vector c_t is the weighted average sum of encoder output
        context = tf.matmul(alignment, encoder_output)

        return context, alignment

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, rnn_size, attention_func):
        super(Decoder, self).__init__()
        self.attention = LuongAttention(rnn_size, attention_func)
        self.rnn_size = rnn_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.lstm = tf.keras.layers.LSTM(rnn_size, return_sequences=True, return_state=True)
        self.wc = tf.keras.layers.Dense(rnn_size, activation='tanh')
        self.ws = tf.keras.layers.Dense(vocab_size)

    def call(self, sequence, state, encoder_output):
        # Remember that the input to the decoder
        # is now a batch of one-word sequences,
        # which means that its shape is (batch_size, 1)
        embed = self.embedding(sequence)

        # Therefore, the lstm_out has shape (batch_size, 1, rnn_size)
        lstm_out, state_h, state_c = self.lstm(embed, initial_state=state)

        # Use self.attention to compute the context and alignment vectors
        # context vector's shape: (batch_size, 1, rnn_size)
        # alignment vector's shape: (batch_size, 1, source_length)
        context, alignment = self.attention(lstm_out, encoder_output)

        # Combine the context vector and the LSTM output
        # Before combined, both have shape of (batch_size, 1, rnn_size),
        # so let's squeeze the axis 1 first
        # After combined, it will have shape of (batch_size, 2 * rnn_size)
        lstm_out = tf.concat(
            [tf.squeeze(context, 1), tf.squeeze(lstm_out, 1)], 1)

        # lstm_out now has shape (batch_size, rnn_size)
        lstm_out = self.wc(lstm_out)

        # Finally, it is converted back to vocabulary space: (batch_size, vocab_size)
        logits = self.ws(lstm_out)

        return logits, state_h, state_c, alignment


# hi_vocab_size = len(hi_tokenizer.word_index) + 1
decoder = Decoder(hi_vocab_size, EMBEDDING_SIZE, RNN_SIZE, ATTENTION_FUNC)

initial_state = encoder.init_states(1)
encoder_outputs = encoder(tf.constant([[1]]), initial_state)
decoder_outputs = decoder(tf.constant([[1]]), encoder_outputs[1:], encoder_outputs[0])

def loss_func(targets, logits):
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, dtype=tf.int64)
    loss = crossentropy(targets, logits, sample_weight=mask)

    return loss


optimizer = tf.keras.optimizers.Adam(clipnorm=5.0)

def predict(test_source_text=None):
    if test_source_text is None:
        test_source_text = lines_en[np.random.choice(len(lines_en))]
    print(test_source_text)
    test_source_seq = en_tokenizer.texts_to_sequences([test_source_text])
    # print(test_source_seq)

    en_initial_states = encoder.init_states(1)
    en_outputs = encoder(tf.constant(test_source_seq), en_initial_states)

    de_input = tf.constant([[hi_tokenizer.word_index['<start>']]])
    de_state_h, de_state_c = en_outputs[1:]
    out_words = []
    alignments = []

    while True:
        de_output, de_state_h, de_state_c, alignment = decoder(de_input, (de_state_h, de_state_c), en_outputs[0])
        de_input = tf.expand_dims(tf.argmax(de_output, -1), 0)
        out_words.append(hi_tokenizer.index_word[de_input.numpy()[0][0]])

        alignments.append(alignment.numpy())

        if out_words[-1] == '<end>' or len(out_words) >= 20:
            break

    print(' '.join(out_words))
    # return np.array(alignments), test_source_text.split(' '), out_words
    return out_words

@tf.function
def train_step(source_seq, target_seq_in, target_seq_out, en_initial_states):
    loss = 0
    with tf.GradientTape() as tape:
        en_outputs = encoder(source_seq, en_initial_states)
        en_states = en_outputs[1:]
        de_state_h, de_state_c = en_states

        # We need to create a loop to iterate through the target sequences
        for i in range(target_seq_out.shape[1]):
            # Input to the decoder must have shape of (batch_size, length)
            # so we need to expand one dimension
            decoder_in = tf.expand_dims(target_seq_in[:, i], 1)
            logit, de_state_h, de_state_c, _ = decoder(decoder_in, (de_state_h, de_state_c), en_outputs[0])

            # The loss is now accumulated through the whole batch
            loss += loss_func(target_seq_out[:, i], logit)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss / target_seq_out.shape[1]

if not os.path.exists('checkpoints_luong/encoder'):
    os.makedirs('checkpoints_luong/encoder')
if not os.path.exists('checkpoints_luong/decoder'):
    os.makedirs('checkpoints_luong/decoder')

# Uncomment these lines for inference mode
encoder_checkpoint = tf.train.latest_checkpoint('checkpoints_luong/encoder')
decoder_checkpoint = tf.train.latest_checkpoint('checkpoints_luong/decoder')

if encoder_checkpoint is not None and decoder_checkpoint is not None:
    encoder.load_weights(encoder_checkpoint)
    decoder.load_weights(decoder_checkpoint)

if MODE == 'train':
    print(NUM_EPOCHS)
    for e in range(NUM_EPOCHS):
        en_initial_states = encoder.init_states(BATCH_SIZE)
        encoder.save_weights('checkpoints_luong/encoder/encoder_{}.h5'.format(e + 1))
        decoder.save_weights('checkpoints_luong/decoder/decoder_{}.h5'.format(e + 1))
        for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
            loss = train_step(source_seq, target_seq_in,target_seq_out, en_initial_states)

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(e + 1, batch, loss.numpy()))

        try:
            predict()

            # predict("How are you today ?")
        except Exception:
            continue

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