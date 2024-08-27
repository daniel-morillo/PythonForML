import tensorflow as tf
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Constantes
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file("iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file("iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
print(train.head())
print(test.head())

train_y = train.pop('Species')
test_y = test.pop('Species')
print(train_y.head())
print(test_y.head())

#Sacamos el total de cada especie para despues compararlo con los datos de nuestro modelo
train_y.value_counts().plot(kind='barh')
plt.show()

#Sacamos el total de cada especie para despues compararlo con los datos de nuestro modelo
test_y.value_counts().plot(kind='barh')
plt.show()

# Normalización de las características
train_stats = train.describe().transpose()

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']  # Normalizamos las características

train_norm = norm(train)
test_norm = norm(test)
print(train_norm.head())
print(test_norm.head())

# Convertimos las etiquetas a one-hot encoding
train_y = tf.keras.utils.to_categorical(train_y)
test_y = tf.keras.utils.to_categorical(test_y)

# Definición del modelo usando Keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(30, activation='relu', input_shape=[len(train.keys())]),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compilamos el modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamos el modelo
model.fit(train_norm, train_y, epochs=200, batch_size=64, validation_split=0.2)

# Evaluamos el modelo en el conjunto de prueba
loss, accuracy = model.evaluate(test_norm, test_y, verbose=2)
print(f"Accuracy: {accuracy}")

# Realizamos predicciones con el modelo
predictions = model.predict(test_norm)

# Convertimos las predicciones y etiquetas de one-hot encoding a clases
predicted_classes = predictions.argmax(axis=1)
true_classes = test_y.argmax(axis=1)

# Calculamos la matriz de confusión
cm = confusion_matrix(true_classes, predicted_classes)

# Visualizamos la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=SPECIES, yticklabels=SPECIES)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()





