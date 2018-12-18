from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM
from keras_contrib.layers import CRF

import pickle
from data import load_data

EMBED_DIM   = 40
BIRNN_UNITS = 50

def model(train=True):
    
    (train_X, train_Y), (dev_X, dev_Y), (vocab, chunks) = load_data()
    
    model = Sequential()
    model.add(Embedding(len(vocab), EMBED_DIM, mask_zero=True))
    model.add(Bidirectional( LSTM(BIRNN_UNITS // 2, return_sequences=True) ))

    crf = CRF(len(chunks), sparse_target=True)
    model.add(crf)
    
    model.summary()
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    if train:
        return model
    
