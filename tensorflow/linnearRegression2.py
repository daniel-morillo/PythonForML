
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc
import tensorflow_estimator as tfe


#Load Data
print()
print()
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

def printData():
    print(dftrain.head(5))
    print(dfeval.head(5))
    print(y_train.head(5))
    print(y_eval.head(5))

def plots():
    dftrain.age.hist(bins=20)
    plt.show()
    dftrain.sex.value_counts().plot(kind='barh')
    plt.show()
    pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
    plt.show()

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']


feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

def make_input(data_df, label_df,epochs = 10, shuffle = True, batchSize = 32):
  def input_function():
    ds = tf.data.Dataset.from_tensor_slices(dict(data_df), label_df)  #Convierte los datos de un pandas Dataset a uno de tensorflow
    #En este caso, transformamos la data_df que son las variables independientes a diccionario, y label_df es lo que se quiere predecir, o la variable dependiente
    if shuffle:
      ds = ds.shuffle(1000)  #En este caso, se mezclan los datos para que el modelo no reconozca patrones que pueden darse debido al orden de los datos
    ds = ds.batch(batchSize).repeat(epochs)  #Se divide el dataset en lotes de batchSize, y se repite el proceso epochs veces
    return ds
  return input_function

train_input_fn = make_input(dftrain, y_train)
eval_input_fn = make_input(dfeval, y_eval, epochs=1, shuffle=False)  #En este caso, con una epoca basta, sino el modelo evaluaria los mismos datos de salida varias veces

#Crear Modelo
linear_est = tfe.estimator.LinearClassifier(feature_columns = feature_columns)


#Entrenar Modelo
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

clear_output()
print("Model Acurracy: " +  str(result['accuracy']))







