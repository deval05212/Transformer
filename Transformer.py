import os
import time
import pickle
import numpy as np

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Dense

class WordPredictor:
    def __init__(self, data_path='data.txt', model_path='next_word_model.keras', tokenizer_path='tokenizer.pkl'):
        self.data_path = data_path
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        
        self.tokenizer = Tokenizer()
        self.model = None
        self.max_len = 0
        self.vocab_size = 0

    def prepare_data(self):
        print("Opening Dataset...")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            text = f.read()

        print("Converting Tokens...")
        self.tokenizer.fit_on_texts([text])
        self.vocab_size = len(self.tokenizer.word_index) + 1
        input_sequences = []

        # Create sequences
        for sentance in text.split('\n'):
            tokenized_sentance = self.tokenizer.texts_to_sequences([sentance])[0]
            for i in range(1, len(tokenized_sentance)):
                n_grams = tokenized_sentance[:i+1]
                input_sequences.append(n_grams)

        self.max_len = max([len(x) for x in input_sequences])

        # Pad_sequences
        pad_input_sequences = pad_sequences(input_sequences, maxlen=self.max_len, padding='pre')

        X = pad_input_sequences[:, :-1]
        y = pad_input_sequences[:, -1]
        y = to_categorical(y, num_classes=self.vocab_size)
        
        return X, y

    def build_and_train(self, epochs=100):
        X, y = self.prepare_data()
        
        output_dim = 100
        input_length = X.shape[1]

        print("Building Model...")
        self.model = Sequential()
        self.model.add(Embedding(input_dim=self.vocab_size, output_dim=output_dim, input_length=input_length))
        self.model.add(LSTM(150))
        self.model.add(Dense(self.vocab_size, activation='softmax'))

        # compile model
        self.model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
        self.model.summary()

        # Train Model
        print(f"Training Model for {epochs} epochs...")
        self.model.fit(X, y, epochs=epochs)

        # Save Model and Tokenizer
        print("Saving Model and Tokenizer...")
        self.model.save(self.model_path)
        with open(self.tokenizer_path, "wb") as f:
            pickle.dump(self.tokenizer, f)

    def load_saved_model(self):
        if os.path.exists(self.model_path) and os.path.exists(self.tokenizer_path):
            print("Loading saved model and tokenizer...")
            self.model = load_model(self.model_path)
            
            with open(self.tokenizer_path, "rb") as f:
                self.tokenizer = pickle.load(f)
                
            # Determine max_len from the model's input shape
            # The model was trained with input_length = max_len - 1
            # So, max_len is model.input_shape[1] + 1
            input_shape = self.model.input_shape[1]
            if input_shape is not None:
                self.max_len = input_shape + 1
            else:
                self.max_len = 20 # fallback if shape is unknown
            
            return True
        return False

    def predict_next_words(self):
        if self.model is None:
            print("Model is not loaded. Cannot predict.")
            return

        print("\n--- Word Prediction (type 'exit' to quit) ---")
        while True:
            text = input("Text: ")
            if text.strip().lower() == 'exit':
                break
            if not text.strip():
                continue
                
            for _ in range(10):
                token_text = self.tokenizer.texts_to_sequences([text])[0]
                # Use max_len - 1 as that is what the model expects as input length
                padded_token_text = pad_sequences([token_text], maxlen=self.max_len - 1, padding='pre')

                # Predict
                predictions = self.model.predict(padded_token_text, verbose=0)
                pos = np.argmax(predictions)

                for word, index in self.tokenizer.word_index.items():
                    if index == pos:
                        text = text + ' ' + word
                        print("Predicted:", text)
                        time.sleep(0.5)
                        break

if __name__ == "__main__":
    predictor = WordPredictor()
    
    print("Starting training process...")
    # This will always train the model and save it.
    # Run this file whenever you update data.txt
    predictor.build_and_train(epochs=100)
    print("Training complete. You can now use prediction.py to test the model.")
