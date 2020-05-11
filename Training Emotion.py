#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 13:47:39 2020

@author: harshita
"""

import numpy as np
import json
from matplotlib import pyplot
from nltk.corpus import stopwords
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
import os
from keras.layers import Embedding,Flatten,Conv1D,MaxPooling1D
from sklearn.preprocessing import MultiLabelBinarizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Embedding,Flatten
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn import metrics

data_file = open(r'/Users/harshita/Desktop/Sentiment_Analysis/nlp_train.json', 'r')
data = json.load(data_file)

x = []
y = [] 

for key in data:
    emotions = []
    x.append(data[key]["body"])
    emotion = data[key]["emotion"]
    for keys in emotion:
        if(emotion[keys]==True):
            emotions.append(keys)
    y.append(emotions)
    
x = np.array(x)
y = np.array(y)    

#print(x.shape)
#print(y.shape)

stop_words = set(stopwords.words('english'))
base_filters = '\n\t!"#$%&()*+,-./:;<=>?[\]^_`{|}~ '
   
word_sequences = []

for i in x:
    i = i.replace('\'','')
    newlist = [x for x in text_to_word_sequence(i,filters=base_filters,lower=True)]
    filtered_sequence = [w for w in newlist if not w in stop_words]
    word_sequences.append(filtered_sequence)

word_sequences = np.array(word_sequences)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(word_sequences)
word_indices = tokenizer.texts_to_sequences(word_sequences)
word_index = tokenizer.word_index

x_data=pad_sequences(word_indices,maxlen=20)

embeddings_index = {}
f = open(os.path.join('', 'glove.6B.50d.txt'),'r',encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((len(word_index) + 1, 50))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,50, weights=[embedding_matrix],input_length=20,trainable=False)


one_hot = MultiLabelBinarizer()
integer_encoded = one_hot.fit_transform(y)
y_data = integer_encoded

#print(y_data)

filename = "model.h5"
if(not os.path.exists(filename)):
    print("Create new model")
    model=Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(30,1,activation="relu"))
    model.add(MaxPooling1D(4))
    model.add(LSTM(100,return_sequences=True))
    model.add(Flatten())
    model.add(Dense(500,activation='relu'))
    model.add(Dense(300,activation='relu'))
    model.add(Dense(y_data.shape[1],activation="sigmoid"))
    model.compile(loss="categorical_crossentropy",optimizer="sgd",metrics=["top_k_categorical_accuracy"])
    print(model.summary())
    
    print("Finished Preprocessing data ...")
    print("x_data shape : ",x_data.shape)
    print("y_data shape : ",y_data.shape)
    
    print("spliting data into training, testing set")
    x_train,y_train = x_data, y_data
    
    batch_size = 128
    num_epochs = 20
    x_valid, y_valid = x_train[:batch_size], y_train[:batch_size]
    x_train2, y_train2 = x_train[batch_size:], y_train[batch_size:]
    
    history=model.fit(x_train2, y_train2, validation_data=(x_valid, y_valid), batch_size=batch_size, epochs=num_epochs)
    print(history)
    print("Saving model")
    model.save(filename, overwrite=True)
else:
    print("Using existing model")
    model = load_model(filename)
    
pyplot.plot(history.history['top_k_categorical_accuracy'],label='Training Accuracy')
pyplot.plot(history.history['val_top_k_categorical_accuracy'],label='Validation Accuracy')

pyplot.legend()
pyplot.show()






