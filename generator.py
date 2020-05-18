from word_predict import WordPredict
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
import random
from sklearn.model_selection import train_test_split

import sys
import os

class Generator:
    def __init__(self, **kwargs):
        self.bs = int(kwargs.get('batch_size'))
        self.filepath = kwargs.get('filepath')
        self.sequences = kwargs.get('sequences')
        self.MAX_NUM_WORDS = kwargs.get('max_words')
        self.MAX_SEQUENCE_LENGTH = kwargs.get('max_len')
        split = kwargs.get('split')
        self.train_sequences, self.test_sequences = train_test_split(self.sequences, test_size=split, random_state=42, shuffle=False)


    def form_sentence_input(self, words):
        #### AICI LOC DE OPTIMIZAT - mutam pad sa fie / batch####
        index = random.randint(0, len(words) - 1)

        no_words = len(words)
        to_predict = words[index]
        to_predict = to_categorical(to_predict, num_classes=self.MAX_NUM_WORDS) ## ca la linia 45 parametrizat

        pre_words = words[:index-1]
        post_words = words[index:]

        pre_words = pad_sequences([pre_words], maxlen=self.MAX_SEQUENCE_LENGTH)[0]
        post_words = pad_sequences([post_words], maxlen=self.MAX_SEQUENCE_LENGTH)[0]

        return (pre_words, post_words, to_predict, no_words)

    def pretrain_gen(self, dataset):
        fill = 0
        x_enc = []
        x_dec = []
        y_dec = []
        if dataset == 'train':
            seq = self.train_sequences
        elif dataset == 'test':
            seq = self.test_sequences
        while True:
            for line in seq:
                if fill < self.bs:
                    (enc_imp, dec_imp, dec_out) = self._noise_input(line, mean_noise=0, var=0)
                    x_enc.append(enc_imp)
                    x_dec.append(to_categorical(dec_imp, num_classes=self.MAX_NUM_WORDS))
                    fill = fill + 1
                else:
                    yield np.array(x_enc), np.array(x_dec)

                    x_enc = []
                    x_dec = []
                    y_dec = []
                    fill = 0

    def next_value_enc_dec(self, dataset):
        fill = 0
        x_enc = []
        x_dec = []
        y_dec = []
        if dataset == 'train':
            seq = self.train_sequences
        elif dataset == 'test':
            seq = self.test_sequences
        while True:
            for line in seq:
                if fill < self.bs:
                    enc_inp = self._noise_input2(line, mean_noise=0.3)
                    for i in range(2,len(line)):
                        x_enc.append(enc_inp)
                        x_dec.append(line[:i])

                        y_dec.append(to_categorical(line[i], num_classes=self.MAX_NUM_WORDS))

                        fill = fill + 1
                else:
                    x_enc = pad_sequences(x_enc, maxlen=self.MAX_SEQUENCE_LENGTH)
                    x_dec = pad_sequences(x_enc, maxlen=self.MAX_SEQUENCE_LENGTH)

                    yield [np.array(x_enc), np.array(x_dec)] \
                        , np.array(y_dec)
                    x_enc = []
                    x_dec = []
                    y_dec = []
                    fill = 0

    def _noise_input2(self, line, mean_noise=0.6, var=0.1):

        noise_size = int(max(np.random.normal(mean_noise, var), 0) * len(line))
        removed_words = np.random.choice(range(len(line)), noise_size)

        enc_inp = line[:]
        for x in removed_words:
            enc_inp[x] = 0

        return enc_inp

    def _noise_input(self, line, mean_noise=0.2, var=0.1):
        dec_inp = line
        noise_size = int(max(np.random.normal(mean_noise, var),0) * len(line))
        removed_words = np.random.choice(range(len(line)), noise_size)

        enc_inp = line[:]
        for x in removed_words:
            enc_inp[x] = 0

        dec_out = to_categorical(pad_sequences([dec_inp[:-1]], maxlen=self.MAX_SEQUENCE_LENGTH)[0],  num_classes=self.MAX_NUM_WORDS)
        enc_inp = pad_sequences([enc_inp], maxlen=self.MAX_SEQUENCE_LENGTH)[0]
        dec_inp = pad_sequences([dec_inp], maxlen=self.MAX_SEQUENCE_LENGTH)[0]

        return np.array(enc_inp), np.array(dec_inp), dec_out

    def generate_enc_dec(self, dataset):
        fill = 0
        x_enc = []
        x_dec = []
        y_dec = []
        if dataset == 'train':
            seq = self.train_sequences
        elif dataset == 'test':
            seq = self.test_sequences
        while True:
            for line in seq:
                if fill < self.bs:
                    (enc_imp, dec_imp, dec_out) = self._noise_input(line)
                    x_enc.append(enc_imp)
                    x_dec.append(dec_imp)

                    y_dec.append(dec_out)
                    fill = fill + 1
                else:
                    yield [np.array(x_enc), np.array(x_dec)]\
                        , np.array(y_dec)
                    x_enc = []
                    x_dec = []
                    y_dec = []
                    fill = 0

    def generate_enc_dec2(self, dataset):
        fill = 0
        x_enc = []
        x_dec = []
        y_dec = []
        if dataset == 'train':
            seq = self.train_sequences
        elif dataset == 'test':
            seq = self.test_sequences
        while True:
            for line in seq:
                if fill < self.bs:
                    (enc_imp, dec_imp, dec_out) = self._noise_input(line)
                    x_enc.append(enc_imp)
                    x_dec.append(to_categorical(dec_imp, num_classes=self.MAX_NUM_WORDS))
                    fill = fill + 1
                else:
                    yield np.array(x_enc), np.array(x_dec)

                    x_enc = []
                    x_dec = []
                    y_dec = []
                    fill = 0


    def generate(self, dataset):
        fill = 0
        y = []
        x0 = []
        x1 = []
        if dataset == 'train':
            seq = self.train_sequences
        elif dataset == 'test':
            seq = self.test_sequences
        while True:
            for line in seq:
                if fill < self.bs:
                    (pre, post, to_predict, _) = self.form_sentence_input(line)

                    x0.append(pre)
                    x1.append(post)
                    y.append(to_predict)
                    fill = fill + 1
                else:

                    yield [np.array(x0), np.array(x1)], np.array(y)
                    x0 = []
                    x1 = []
                    y = []
                    fill = 0

if __name__ == '__main__':
    gen = Generator(filepath='all.txt', batch_size=32)
    for x, y in gen.generate():

        print(y)

        break
