from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

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
            y.append(FEMALE)
	
    return data, np.array(y)


def keras_model():

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(32, activation=tf.nn.elu, input_shape=(20,)))
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.elu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.elu))
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.elu))
    model.add(tf.keras.layers.Dense(32, activation=tf.nn.elu))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    return model


#model = keras_model()
model = tf.keras.models.load_model('ia.model')#= keras_model()

model.compile(
    loss=tf.keras.losses.mean_squared_error,
    optimizer=tf.keras.optimizers.SGD(1e-2),
    metrics=['accuracy']
)
features, labels = get_data()
model.fit(
    features,
    labels,
    epochs=1000,
    shuffle=True,
    batch_size=5

)

model.save('ia.model')
