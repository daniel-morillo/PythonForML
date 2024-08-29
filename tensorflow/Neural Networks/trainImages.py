import tensorflow as tf
from tensorflow import keras

import numpy as np 
import matplotlib.pyplot as plt 

fashionMnist = keras.datasets.fashion_mnist

(trainImages, trainLabels) , (testImages, testLabels) = fashionMnist.load_data()

print(trainImages.shape)

#Our labels are integers ranging from 0 - 9. Each integer represents a specific article of clothing. We'll create an array of label names to indicate which is which.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Plotting the images
plt.figure()
plt.imshow(trainImages[1])
plt.colorbar()
plt.grid(False)
#plt.show()

#Preprocessing Data
#Here I'm going to divide all of our pixels to be in a range between 0 and 1
trainImages = trainImages / 255.0
testImages = testImages / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax')
])
 
model.compile(optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'])

model.fit(trainImages,trainLabels, epochs = 10)

testLoss, testAccuracy = model.evaluate(testImages,testLabels,verbose = 1)

print('Test Accuracy: ', testAccuracy)

predictions = model.predict(testImages)

for i, prediction in enumerate(predictions):
    if i > 100:
        break
    print(f"Image {i}: Predicted class {np.argmax(prediction)}")

COLOR = 'blue'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  prediction = model.predict(np.array([image]))
  predicted_class = class_names[np.argmax(prediction)]

  show_image(image, class_names[correct_label], predicted_class)


def show_image(img, label, guess):
  plt.figure()
  plt.imshow(img, cmap=plt.cm.binary)
  plt.title("Excpected: " + label)
  plt.xlabel("Guess: " + guess)
  plt.colorbar()
  plt.grid(False)
  plt.show()


def get_number():
  while True:
    num = input("Pick a number: ")
    if num.isdigit():
      num = int(num)
      if 0 <= num <= 1000:
        return int(num)
    else:
      print("Try again...")

num = get_number()
image = testImages[num]
label = testLabels[num]
predict(model, image, label)



