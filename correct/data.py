import numpy as np

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import pickle


CORPUS_PATH = "/home/lijiazheng/corpus/correct/"

def load_data():
    with open(CORPUS_PATH + "chars_simple.dict") as f:
        chars = [line.strip() for line in f]
    with open(CORPUS_PATH + "pinyin.dict") as f:
        pinyins = [line.strip() for line in f]

    with open(CORPUS_PATH + "pinyin-test.txt") as f:#"pinyin-stock-corpus.txt") as f:
        datas = [line.strip().split("\t")[:2] for line in f]
    
    X, Y = _process_data(datas, pinyins, chars)
    train_X, dev_X, train_Y, dev_Y = train_test_split(X, Y)
    
    return (train_X, train_Y), (dev_X, dev_Y), (pinyins, chars)


def data_generator():
    with open(CORPUS_PATH + "chars_simple.dict") as f:
        chars = [line.strip() for line in f]
     with open(CORPUS_PATH + "pinyin.dict") as f:
        pinyins = [line.strip() for line in f]
    
    with open(CORPUS_PATH + "pinyin-stock-corpus.txt") as f1, open(CORPUS_PATH + "pinyin-no-stock-corpus.txt") as f2:
        for _ in range(BATCH_SIZE):


    with open(CORPUS_PATH + "pinyin-test.txt") as f:#"pinyin-stock-corpus.txt") as f:
        datas = [line.strip().split("\t")[:2] for line in f]
    

def _process_data(data, pinyins, chars, max_len=10):
    pinyin2idx = dict((w, i) for i, w in enumerate(pinyins))
    word2idx = dict((w, i) for i, w in enumerate(chars))
    x = [[pinyin2idx.get(w, 0) for w in s[0].split(" ")] for s in data]
    y = [[word2idx.get(w, 0) for w in s[1].split(" ")] for s in data]

    x = pad_sequences(x, max_len, padding="post")
    y = pad_sequences(y, max_len, padding="post")
    print(data[0], x[0], y[0])
    
    y = np.expand_dims(y, 2)
    
    return x, y


if __name__ == "__main__":
    print(load_data()[:2])
