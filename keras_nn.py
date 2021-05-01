import pandas as pd
import numpy as np
import nltk
import sklearn
import torch

from nltk.stem import WordNetLemmatizer
from torch.utils.data.dataset import Dataset
from data import *

import sklearn.model_selection as model_selection
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import keras
import tensorflow as tf
from keras import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Activation

vocab_size = 1000
training_size = 0.7

def get_data():

    preprocessed_lyrics, genre = data.get_dataset()
    num_data = len(preprocessed_lyrics)
    X_train, X_test = preprocessed_lyrics[:int(num_data * training_size)], preprocessed_lyrics[int(num_data * training_size):]
    y_train, y_test = genre[:int(num_data * training_size)], genre[int(num_data * training_size):]

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    return X_train, X_test, y_train, y_test

def simple_model():
    model = Sequential()
    model.add(Embedding(1000, 32, input_length=200, mask_zero=True))
    model.add(LSTM(32))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=7, activation='softmax'))
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer="adam", metrics=['acc'])
    return model

def bidirectional_model():
    return None

model = simple_model()

X_train, X_test, y_train, y_test = get_data()

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#print(type(X_train))
#print(type(y_train))

model.fit(X_train, y_train, epochs=20)
print("Accuracy: ", model.evaluate(X_test, y_test)[1])
