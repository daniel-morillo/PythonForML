import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import os
import numpy as np 
keras = tf.keras

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

#Normalize
train_images = train_images / 255.0
test_images = test_images / 255.0

#Define class names to be used
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

IMG_INDEX = 16

def augmentateData(train_images):
    #Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    #Pick an Image to transform
    test_img = train_images[IMG_INDEX]
    img = image.img_to_array(test_img)
    img = img.reshape((1,) + img.shape)

    i = 0

    for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):  # this loops runs forever until we break, saving images to current directory with specified prefix
        plt.figure(i)
        plot = plt.imshow(image.img_to_array(batch[0]))
        i += 1
        if i > 4:  # show 4 images
            break

    plt.show()






#Plotting Image
def showImage(IMG_INDEX, train_images, train_labels, class_names):
    plt.imshow(train_images[IMG_INDEX], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
    plt.show()



#Create the Convolutional Base
def createModelBase():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    return model

model = createModelBase()

#Add Dense Layers on top
#This is the classifier of our model
def addModelClassifier(model):
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    return model

model = addModelClassifier(model)

def showModelSummary(model):
    print(model.summary())

showModelSummary(model)

#Compile and train the model
def trainModel(model, train_images, train_labels, test_images, test_labels):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=4,
                        validation_data=(test_images, test_labels)) #Validation data is used to test the model


def usePretrainedModel():
    (raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
    )

    get_label_name = metadata.features['label'].int2str

    #Displaying 2 images from the dataset
    def displayImage(raw_train):
        for image, label in raw_train.take(5):
            plt.figure()
            plt.imshow(image)
            plt.title(get_label_name(label))
    
    displayImage(raw_train)




trainModel(model, train_images, train_labels, test_images, test_labels)

#Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
print(test_accuracy)

#Make predictions
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

#Plotting the first 25 test images, their predicted label, and the true label
def plotImages(predictions, test_labels, test_images):
    num_rows = 5
    num_cols = 5
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plt.bar(range(10), predictions[i])
        plt.xticks(range(10))
        plt.yticks([])
        predicted_label = np.argmax(predictions[i])
        true_label = test_labels[i]
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'
        #plt.xlabel("{} ({})".format(class_names[predicted_label],
                                    #class_names[true_label]),
                                    #color=color)
    plt.tight_layout()
    plt.show()

plotImages(predictions, test_labels, test_images)



