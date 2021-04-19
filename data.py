import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

dataset = pd.read_csv("lyrics_data.csv")
genre = dataset['genre']
lyrics = dataset['lyrics']

genre_values = dict() # contains how many of each genre there are
for index, value in genre.items():
    if value not in genre_values:
        genre_values[value] = 0
    genre_values[value] += 1

genre = pd.get_dummies(genre) # One-hot encode

# Function to lemmatize lyrics - NOTE: The data already seems pretty lemmatized
def lemmatize(data):
    data = data.str.split(" |,|<br /><br />|\"|!")
    lemmatizer = WordNetLemmatizer()

    for i in range(len(data)):
        data.iloc[i] = [lemmatizer.lemmatize(word) for word in data.iloc[i]]
    
    return data

lyrics = lemmatize(lyrics) # lemmatize lyrics

# Returns key of dictionary with largest value
def key_of_max_val(data):
    max_key = 0
    max_val = 0
    for key in data.keys():
        if data[key] > max_val:
            max_key = key
            max_val = data[key]
    
    return max_key

vocab_size = 1000

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

vocab_dict = create_frequency_dict(lyrics) # maps word to relative frequency

# 0 is padding, 1 is start, 2 is unknown word, 3 and above is word value
def get_word_value(word):
    if word not in vocab_dict.keys():
        return 2
    return vocab_dict[word]

max_words = 0 # Longest lyrics
for i in range(len(lyrics)):
    if len(lyrics.iloc[i]) > max_words:
        max_words = len(lyrics.iloc[i])

# Replaces words with frequency values
preprocessed_lyrics = []
for i in range(len(lyrics)):
    song = []
    for _ in range(max_words - len(lyrics.iloc[i])):
        song.append(0)
    song.append(1)
    for word in lyrics.iloc[i]:
        song.append(get_word_value(word))
    preprocessed_lyrics.append(song)
