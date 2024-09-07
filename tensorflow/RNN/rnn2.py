from keras.preprocessing import sequence
import tensorflow as tf
import keras
import os
import numpy as np
from tensorflow.keras.models import load_model

# Downloading file
def getFile(filename, URL):
    path_to_file = tf.keras.utils.get_file(filename, URL)
    return path_to_file

def fileToText(filename):
    text = open(filename, 'rb').read().decode(encoding='utf-8')
    return text

# Encoding
def encodeText(text):
    vocab = sorted(set(text))
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    return np.array([char2idx[c] for c in text]), idx2char, char2idx, vocab

def decodetext(integers, indx2char):
    try:
        integers = np.array(integers)
    except:
        pass
    return ''.join(indx2char[integers])

def createSequences(text_as_int, seqLength):
    charDataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = charDataset.batch(seqLength + 1, drop_remainder=True)
    return sequences

def splitInputTarget(chunk):
    inputText = chunk[:-1]
    targetText = chunk[1:]
    return inputText, targetText

# Create Model
def createModel(vocab_size, embeddingDim, rnnUnits, batchSize):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(batch_input_shape=[batchSize, None]),  # Stateful model
        tf.keras.layers.Embedding(vocab_size, embeddingDim),
        tf.keras.layers.LSTM(rnnUnits, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

# Custom loss function
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def trainModel(model, data, epochs):
    history = model.fit(data, epochs=epochs)
    return history

# Function to generate text
def generate_text(model, start_string, char2idx, indx2char):
    num_generate = 800
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    temperature = 1.0
    model.layers[1].reset_states()
    
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(indx2char[predicted_id])
    
    return start_string + ''.join(text_generated)


# Load the model and pass the custom loss function
try:
    model = load_model('text_generator_model.h5', custom_objects={'loss': loss})
except:
    model = None
path_to_file = getFile('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = fileToText(path_to_file)
text_as_int, indx2char, char2idx, vocab = encodeText(text)

if model is not None:
    print('Model loaded successfully')
    print(model.summary())
else:
    print('Model not found')

    print(f'Length of Text: {len(text)}')
    print(f'Encoded Text: {text_as_int[:13]}')
    print(f'Decoded Text: {decodetext(text_as_int[:13], indx2char)}')

    # Our training examples are going to be a sentence for input and the same sentence for output but shifted one character to the right
    seqLength = 100
    examplesPerEpoch = len(text) / (seqLength + 1)

    sequences = createSequences(text_as_int, seqLength)
    dataset = sequences.map(splitInputTarget)

    BATCH_SIZE = 64
    VOCAB_SIZE = len(vocab)
    EMBEDDING_DIM = 256
    RNN_UNITS = 1024
    BUFFER_SIZE = 10000

    data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    model = createModel(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
    print(model.summary())

    # Compiling model
    model.compile(optimizer='adam', loss=loss)

    # Training the model
    EPOCHS = 2
    history = trainModel(model, data, EPOCHS)

    #saveModel
    model.save('text_generator_model.h5')

# Generate text
print(generate_text(model, start_string=u"ROMEO: ", char2idx=char2idx, indx2char=indx2char))


















