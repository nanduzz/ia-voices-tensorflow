from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing

LABEL_COL = 20
MALE = [1, 0]
FEMALE = [0, 1]

def get_data():
    data = np.array(pd.read_csv('voice.csv'))
    shuffle(data)
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

    model.add(tf.keras.layers.Dense(
        32, activation=tf.nn.elu, input_shape=(20,)))
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.elu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.elu))
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.elu))
    model.add(tf.keras.layers.Dense(32, activation=tf.nn.elu))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    return model


model = keras_model()
#model = tf.keras.models.load_model('ia.model')

model.compile(
    loss=tf.keras.losses.mean_squared_error,
    optimizer=tf.keras.optimizers.SGD(1e-2),
    metrics=['accuracy'],
)

features, labels = get_data()
features = preprocessing.MinMaxScaler().fit_transform(features)

train_features = features[:-500]
test_features = features[-500:]
train_labels = labels[:-500]
test_labels = labels[-500:]


tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs/tensorboard', histogram_freq=0,
                                             write_graph=True, write_images=False)

model.fit(
    features,
    labels,
    epochs=500,
    shuffle=True,
    batch_size=5,
    callbacks=[tensorboard],
    validation_split=0.2
)

#model.save('ia.model')
