from keras.datasets import imdb
from keras.preprocessing import sequence
import tensorflow as tf
import os
import numpy as np

VOCAB_SIZE = 88584

MAXLEN = 250
BATCH_SIZE = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)

def printWordsFrequency(train_data):
    wordsFrequency = {}
    for wordEncoded in train_data[1:]:
       for word in wordEncoded:
           if word in wordsFrequency:
               wordsFrequency[word] += 1
           else:
               wordsFrequency[word] = 1

    print(wordsFrequency)

#Padding Data
#Here we need to pad the training and test data, so all data have the same length
#If the data has more than 250 words, we need to trim the extra words. 
#If the data has less than 250 words, we need to add zeros to it.

train_data = sequence.pad_sequences(train_data, MAXLEN)
test_data = sequence.pad_sequences(test_data, MAXLEN)

#Creating Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),  #32 is the output dimension of the embedding layer or vectors
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation='sigmoid') #Here we are classifying the data, so we use sigmoid, 0 is a negative review and 1 a positive one
])

#Training
model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['acc'])

history = model.fit(train_data, train_labels, epochs = 10, validation_split = 0.2)

#Evaluating the model
results = model.evaluate(test_data, test_labels)
print(results)