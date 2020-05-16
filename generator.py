from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import numpy as np
import random
import sys
import os
import time
class Generator:
    def __init__(self, **kwargs):
        self.bs = int(kwargs.get('batch_size'))
        self.filepath = kwargs.get('filepath')
        self.texts = []
        self.init_tokenizer()

    def form_sentence_input(self, words):
        #### AICI LOC DE OPTIMIZAT - mutam pad sa fie / batch####
        index = random.randint(0, len(words) - 1)

        no_words = len(words)
        to_predict = words[index]
        to_predict = to_categorical(to_predict, 20000) ## ca la linia 45 parametrizat

        pre_words = words[:index-1]
        post_words = words[index:]

        pre_words = pad_sequences([pre_words], maxlen=100)[0]
        post_words = pad_sequences([post_words], maxlen=100)[0]


        return (pre_words, post_words, to_predict, no_words)

    def form_input(self, sentences, bs=0):

        return self.form_sentence_input(sentences)

    def init_tokenizer(self):
        with open(self.filepath, encoding='utf-8') as fp:
            for line in fp:
                if line == "\n":
                    continue
                self.texts.append(line)

        self.tokenizer = Tokenizer(num_words=20000) #params
        self.tokenizer.fit_on_texts(self.texts)
        self.sequences = self.tokenizer.texts_to_sequences(self.texts)
        self.word_index = self.tokenizer.word_index
        print('Found %s unique tokens.' % len(self.word_index))



    def generate(self):
        if not os.path.isfile(self.filepath):
            print("File path {} does not exist. Exiting...".format(self.filepath))
            sys.exit()


        fill = 0
        x = []
        y = []
        for line in self.sequences:
            if fill < self.bs:
                if len(line) > 5: # param
                    (pre, post, to_predict, _) = self.form_sentence_input(line)

                    xi = []

                    xi.append(pre)
                    xi.append(post)

                    x.append(xi)
                    y.append(to_predict)
                    fill = fill + 1
            else:

                yield x, y
                x = []
                y = []
                fill = 0


if __name__ == '__main__':
    gen = Generator(filepath='all.txt', batch_size=32)
    for x, y in gen.generate():

        print(y)

        break
