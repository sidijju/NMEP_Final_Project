import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from torch.utils.data.dataset import Dataset
import sklearn
import torch

vocab_size = 1000

# Function to lemmatize lyrics - NOTE: The data already seems pretty lemmatized
def lemmatize(data):
    data = data.str.split(" |,|<br /><br />|\"|!")
    lemmatizer = WordNetLemmatizer()

    for i in range(len(data)):
        data.iloc[i] = [lemmatizer.lemmatize(word) for word in data.iloc[i]]

    return data

# Returns key of dictionary with largest value
def key_of_max_val(data):
    max_key = 0
    max_val = 0
    for key in data.keys():
        if data[key] > max_val:
            max_key = key
            max_val = data[key]

    return max_key

# Returns dictionary mapping each word to relative frequency
def create_frequency_dict(data):
    freq = dict()
    for song in data:
        for word in song:
            if word not in freq:
                freq[word] = 0
            freq[word] += 1

    vocab_dict = dict()
    for i in range(3, vocab_size):
        key = key_of_max_val(freq)
        vocab_dict[key] = i
        freq[key] = 0

    return vocab_dict

# 0 is padding, 1 is start, 2 is unknown word, 3 and above is word value
def get_word_value(word, vocab_dict):
    if word not in vocab_dict.keys():
        return 2
    return vocab_dict[word]

def get_max_words(data):
    max_words = 0 # Longest lyrics
    for i in range(len(data)):
        if len(data.iloc[i]) > max_words:
            max_words = len(data.iloc[i])

    return max_words

def get_dataset():
    # Uncomment following line if its the first time calling this program
    #nltk.download('wordnet')

    dataset = pd.read_csv("lyrics_data.csv")
    dataset = dataset.sample(frac=1).reset_index(drop=True) # shuffle
    genre = dataset['genre']#.iloc[:1000] # TODO: Change back
    lyrics = dataset['lyrics']#.iloc[:1000] # TODO: Change back

    #print(type(genre))
    #print(type(lyrics))

    genre_values = dict() # contains how many of each genre there are
    genre_indicies = dict()
    for index, value in genre.items():
        if value not in genre_values:
            genre_indicies[value] = len(genre_values)
            genre_values[value] = 0

        genre_values[value] += 1

    # One-hot encode
    genre = pd.get_dummies(genre)

    #print(genre_indicies)
    #genre_labels = []
    #for i in range(len(genre)):
    #    genre_labels.append(genre_indicies[genre[i]])

    # lemmatize lyrics
    lyrics = lemmatize(lyrics)

    # get max words
    max_words = get_max_words(lyrics)

    # maps word to relative frequency
    vocab_dict = create_frequency_dict(lyrics)

    # Replaces words with frequency values
    preprocessed_lyrics = []
    for i in range(len(lyrics)):
        song = []
        for _ in range(max_words - len(lyrics.iloc[i])):
            song.append(0)
        song.append(1)
        for word in lyrics.iloc[i]:
            song.append(get_word_value(word, vocab_dict))
        preprocessed_lyrics.append(song)

    #print(genre_labels)
    #preprocessed_lyrics, genre_labels = sklearn.utils.shuffle(preprocessed_lyrics, genre_labels)

    #print(np.array(preprocessed_lyrics).shape)
    #print(np.array(genre_labels).shape)
    #print(genre_labels)

    return (preprocessed_lyrics, genre)

preprocessed_lyrics, genre = get_dataset()

from sklearn import svm, datasets
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

#print(len(preprocessed_lyrics))
#print(len(genre_labels))

training_size = 0.75
num_data = len(preprocessed_lyrics)
X_train, X_test, y_train, y_test = preprocessed_lyrics[:int(num_data * training_size)], preprocessed_lyrics[int(num_data * training_size):], genre[:int(num_data * training_size)], genre[int(num_data * training_size):]

import keras
from keras import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Activation, Bidirectional, Dropout

def simple_model():
    model = Sequential()
    model.add(Embedding(1000, 32, input_length=200, mask_zero=True))
    model.add(LSTM(32))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=7, activation='softmax'))
    return model

def stacked_model():
    model = Sequential()
    model.add(Embedding(1000, 32, input_length=200, mask_zero=True))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dropout(0.5))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=7, activation='softmax'))
    return model

def bidirectional_model():
    model = Sequential()
    model.add(Embedding(1000, 32, input_length=200, mask_zero=True))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.5))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=7, activation='softmax'))
    return model

def stacked_bidirectional_model():
    model = Sequential()
    model.add(Embedding(1000, 32, input_length=200, mask_zero=True))
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.5))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=7, activation='softmax'))
    return model


model = simple_stacked_model()

import tensorflow as tf
model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer="adam", metrics=['acc'])

#print(type(X_train))
#print(type(y_train))

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

#print(X_train.shape)
#print(y_train.shape)
#print(X_test.shape)
#print(y_test.shape)

model.fit(X_train, y_train, validation_split = 0.1, shuffle = True, epochs=50)
print("Accuracy: ", model.evaluate(X_test, y_test)[1])
