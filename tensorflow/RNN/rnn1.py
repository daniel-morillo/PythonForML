from keras.datasets import imdb
from keras.preprocessing import sequence
import tensorflow as tf
import os
import numpy as np
import keras.preprocessing 

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

def padData(train_data, test_data):
    train_data = sequence.pad_sequences(train_data, MAXLEN)
    test_data = sequence.pad_sequences(test_data, MAXLEN)
    return train_data, test_data

train_data, test_data = padData(train_data, test_data)

#Creating Model
def createModel():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, 32),  #32 is the output dimension of the embedding layer or vectors
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1, activation='sigmoid') #Here we are classifying the data, so we use sigmoid, 0 is a negative review and 1 a positive one
    ])
    return model

model = createModel()

def trainModel(model):
    #Training
    model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['acc'])
    
    history = model.fit(train_data, train_labels, epochs = 5, validation_split = 0.2)

    return history

history = trainModel(model)


#Evaluating the model
def evaluateModel(model):
    results = model.evaluate(test_data, test_labels)
    print(results)

evaluateModel(model)


#Encoding text
word_index = imdb.get_word_index()
#print(word_index)

def encodeText(text):

    tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return sequence.pad_sequences([tokens], MAXLEN)[0]


#decoding
reverse_word_index = {value: key for (key, value) in word_index.items()}

def decodeIntegers(integers):
    PAD = 0
    text = ""
    for num in integers:
        if num != PAD:
            text += reverse_word_index[num] + " "

    return text[:-1]  #Se corta el ultimo caracter de la cadena que es un espacio en blanco

def predict(model, text):
    encodedText = encodeText(text)
    prediction = np.zeros((1, 250))
    prediction[0] = encodedText
    result = model.predict(prediction)
    if result[0] >= 0.5:
        print("Positive Review")
    else:
        print("Negative Review")

positive_review = "That movie was! really loved it and would great watch it again because it was amazingly great"
predict(model,positive_review)

negative_review = "that movie really sucked. I hated it and wouldn't watch it again. Was one of the worst things I've ever watched"
predict(model,negative_review)

    