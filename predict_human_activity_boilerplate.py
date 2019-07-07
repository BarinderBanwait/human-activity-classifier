"""predict_human_activity_boilerplate 

--------------------------------------92% ACCURACY----------------------------------------------

Sensor data is loaded from dataframes.py, and then a dNN is created, compiled, and
trained on it. It is evaluated against a test set, and the accuracy score is output.

This runs the standard parameters found on the tensorflow tutorial. Accuracy score is 92%

It might be possible to get a higher score by doing the following:

1. Change batch_size
2. Change which columns are used as features (e.g. use fewer columns, use principal components)
3. Change parameters in model creation and compilation

That last one requires an understanding of what is actually going on with neural networks. A good 
place to learn that stuff is the Google online Machine Learning Course. Have fun with it!!!

NB. If you want to change the parameters, please do so on a copy; leave this file alone as a 
record of the standard boilerplate parameters and score.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import tensorflow as tf

from tensorflow import feature_column
from tensorflow import keras
from tensorflow.keras import layers
import re
print(tf.__version__)

from dataframes import train, val, test, activity_labels_dict

print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

# Create an input pipeline using tf.data

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

feature_columns = []

# numeric cols
for header in train.columns[:-1]: # Customise which columns you want here
  feature_columns.append(feature_column.numeric_column(header))

# Create feature layer

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# Create batched tf datasets

batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# Create, compile, and train the model

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(len(activity_labels_dict), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              run_eagerly=True)

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)

