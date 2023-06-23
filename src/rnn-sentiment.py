from keras.datasets import imdb
from keras.preprocessing import process
import tensorflow as tf
import os
import numpy as numpy

VOCAB_SIZE = 88584
MAX_LEN = 250

(trainData, trainLabels), (evalData, evalLabels) = imbdb.load_data(num_words=VOCAB_SIZE)

trainData = process.pad_sequences(trainData, MAX_LEN)
evalData = process.pad_sequences(evalData, MAX_LEN)

rnnModel = tf.keras.sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

rnnModel.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['acc'])
history = model.fit(trainData, trainLabels, epochs=10, validation_split=0.2)

results = rnnModel.evaluate(evalData, evalLabels)

wordIndex = imdb.get_word_index()
reverseWordIndex = {value: key for (key, value) in wordIndex.items()}



#Used to encode your own reviews to have processed in the model
def encode_text(funcText):
    tokens = process.text.text_to_word_sequence(funcText)
    tokens = [wordIndex[word] if word in word_index else 0 for word in tokens]
    return process.pad_sequences([tokens], MAX_LEN)[0]

#Used for turning encoded text back into something readable
def decode_text(funcEncodedText):
    text = ""
    for word in funcEncodedText:
        if word != 0:
            text += reverseWordIndex[word] + " "
    return text[:-1]

#Makes prediction based on input text using our model
def predict_text(funcText):
    encodedText = encode_text(funcText)
    prediction = np.zeros((1,250))
    prediction[0] = encodedText
    result = rnnModel.predict(prediction)
    return result
