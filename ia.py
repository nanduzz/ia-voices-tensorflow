import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import shuffle from random

LABEL_COL = 20
MALE = [1,0]
FEMALE = [0,1]


def get_data():
    data = np.array(pd.read_csv('voice.csv'))
    labels = data[:, [LABEL_COL]]
    data = np.delete(data, LABEL_COL, axis=1)

    y = []
    for label in labels:
        if label == 'male':
            y.append(MALE)
        else:
            y.append(FAMALE)
	
    return data, np.array(y)


def keras_model():

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(32, input_shape=(20,)))
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Dense(32))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    return model


#model = keras_model()
model = tf.keras.models.load_model('ia.model')#= keras_model()

model.compile(
    loss=tf.keras.losses.mean_squared_error,
    optimizer=tf.keras.optimizers.SGD(1e-5),
    metrics=['accuracy']
)
features, labels = get_data()
model.fit(
    features,
    labels,
    epochs=1000,
    shuffle=True,
    batch_size=1

)

model.save('ia.model')
