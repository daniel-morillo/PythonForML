#We are going to predict the next character of a text
#We are going to use the Shakespeare dataset
from keras.preprocessing import sequence
import tensorflow as tf
import keras
import os
import numpy as np

#Downloading file
def getFile(filename, URL):
    path_to_file = tf.keras.utils.get_file(filename, URL)
    return path_to_file

path_to_file = getFile('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

def fileToText(filename):
    text = open(filename, 'rb').read().decode(encoding='utf-8')
    return text

text = fileToText(path_to_file)
print(f'Length of Text: {len(text)}')

#Encoding
def encodeText(text):
    vocab = sorted(set(text))
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    return np.array([char2idx[c] for c in text]), idx2char, char2idx, vocab



text_as_int, indx2char,char2idx , vocab = encodeText(text)

def decodetext(integers,indx2char):
    try:
        integers = np.array(integers)
    except:
        pass
    return ''.join(indx2char[integers])

print(f'Encoded Text: {text_as_int[:13]}')
print(f'Decoded Text: {decodetext(text_as_int[:13],indx2char)}')

#Our training examples ar going to be a sentence for input and the same sentence for output but shifted one character to the right

seqLength = 100
examplesPerEpoch = len(text)/(seqLength+1)

def createSequences(text_as_int, seqLength):
    charDataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    sequences = charDataset.batch(seqLength+1, drop_remainder = True)
    return sequences

sequences = createSequences(text_as_int, seqLength)

def splitInputTarget(chunk):
    inputText = chunk[:-1]
    targetText = chunk[1:]
    return inputText, targetText

dataset = sequences.map(splitInputTarget)

#for example:
def printExample(dataset):
    for inputExample, targetExample in dataset.take(1):
        print("Input: " , decodetext(inputExample,indx2char))
        print("Target: ", decodetext(targetExample,indx2char))

BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 256
RNN_UNITS = 1024

BUFFER_SIZE = 10000

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder = True)

#Create Model
def createModel(vocab_size, embeddingDim, rnnUnits, batchSize):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(batch_input_shape = [batchSize, None]),  #We need to pass a InputLayer in Stateful Models
        tf.keras.layers.Embedding(vocab_size, embeddingDim),
        tf.keras.layers.LSTM(rnnUnits, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


model = createModel(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
print(model.summary())


def lookingModelOutput(data, model):
    for input_example_batch, target_example_batch in data.take(1):
        example_batch_predictions = model(input_example_batch)  # ask our model for a prediction on our first batch of training data (64 entries)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")  # print out the output shape

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

#Compiling model
model.compile(optimizer = 'adam', loss = loss)

