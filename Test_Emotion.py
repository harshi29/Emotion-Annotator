#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:37:29 2020

@author: harshita
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 13:47:39 2020

@author: harshita
"""

import numpy as np
import json
from nltk.corpus import stopwords
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
import os
from keras.layers import Embedding
from sklearn.preprocessing import MultiLabelBinarizer
from keras.models import load_model
from sklearn import metrics


test_file = open(r'/Users/harshita/Desktop/Sentiment_Analysis/nlp_test.json', 'r')

test = json.load(test_file)


x_test = []
y_test = []
 
for key in test:
    emotions = []
    x_test.append(test[key]["body"])
    emotion = test[key]["emotion"]
    for keys in emotion:
        if(emotion[keys]==True):
            emotions.append(keys)
    y_test.append(emotions)

x_test = np.array(x_test)
y_test = np.array(y_test)  

#print(x.shape)
#print(y.shape)

stop_words = set(stopwords.words('english'))
base_filters = '\n\t!"#$%&()*+,-./:;<=>?[\]^_`{|}~ '
   
word_sequences = []

for i in x_test:
    i = i.replace('\'','')
    newlist = [x for x in text_to_word_sequence(i,filters=base_filters,lower=True)]
    filtered_sequence = [w for w in newlist if not w in stop_words]
    word_sequences.append(filtered_sequence)

word_sequences = np.array(word_sequences)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(word_sequences)
word_indices = tokenizer.texts_to_sequences(word_sequences)
word_index = tokenizer.word_index
x_test=pad_sequences(word_indices,maxlen=20)


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
integer_encoded = one_hot.fit_transform(y_test)
y_test = integer_encoded

filename = "model.h5"
model = load_model(filename)
    
y_pred= model.predict(x_test)
y_pred= (y_pred>0.7).astype(int)
acc= np.mean(y_pred==y_test)
print('Total test accuracy is:', acc)

accuracy = []
f1_scores = []
for i in range(len(y_test[0])):
    accuracy.append(metrics.accuracy_score(y_test[:,i],y_pred[:,i]))
    f1_scores.append(metrics.f1_score(y_test[:,i],y_pred[:,i]))
#print(accuracy)
    


print("Emotion-wise test accuracy")
classes= one_hot.classes_
for label, acc in zip(classes, accuracy):
    print(label, acc)

print('\n')
for label, f1_score in zip(classes, f1_scores):
    print(label, f1_score)
