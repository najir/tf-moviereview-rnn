# tf-moviereview-rnn
    Isaac Perks
    06-22-2023

# Description

Using IMDB's Movie Review dataset from keras on a basic RNN to create sentiment analysis based on whether the review was positive, negative, or neutral

- Import IMDB dataset from keras
    - Load data with datasets set Vocab Size(saved as a constant based on dataset)
    - Pad the sequences so they are all of length 250(set by maxlength constant)
- Build our model architecture
    - An embedding layer with our vocab size and an output dimension of 32
    - Set up an LSTM layer of 32
    - Finally a dense layer of 1 and a sigmoid activation
- Compile the model and fit the data
    - epochs 10 and a validation split of 0.2
    - rmsprop optimizer
    - binary crossentropy
- Supporting functions
    - An encoding function to encode input text with the current vocab list for model use
    - A decode function to view existing texts or words
    - A prediction function that takes text input, encodes it, and predicts it with our model