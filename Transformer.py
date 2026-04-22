import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.models import Sequential, save_model
from keras.layers import Embedding, LSTM, Dense

import pickle
import numpy as np

tokenizer = Tokenizer()

with open('data.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# print(text)
tokenizer.fit_on_texts([text])
tokens_len = len(tokenizer.word_index)
vocab_size = tokens_len + 1
input_sequences = []

for sentances in text.split('\n'):
    tokenized_sentance = tokenizer.texts_to_sequences([sentances])[0]

    for i in range(1,len(tokenized_sentance)):
        n_grams = tokenized_sentance[:i+1]
        input_sequences.append(n_grams)

max_len = max([len(x) for x in input_sequences])

pad_input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

X = pad_input_sequences[:,:-1]
y = pad_input_sequences[:,-1]
num_classes =len(set(y))
y = to_categorical(y, num_classes=vocab_size)

output_dim = 100
input_length = X.shape[1]

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=output_dim, input_length=input_length))
model.add(LSTM(150))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

model.summary()


model.fit(X,y, epochs=100)

model.save("next_word_model.keras")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

import time 

for i in range(5):
    # Prediction
    text = input("Text: ")

# tokenize text
    token_text = tokenizer.texts_to_sequences([text])[0]
    padded_token_text = pad_sequences([token_text], maxlen=max_len, padding='pre')


    pos = np.argmax(model.predict(padded_token_text))

    for word,index in tokenizer.word_index.items():
        if index == pos:
            text = text + ' ' + word
            print(text)
            time.sleep(2)