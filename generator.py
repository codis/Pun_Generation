from word_predict import WordPredict
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
import random
import sys
import os

class Generator:
    def __init__(self, **kwargs):
        self.bs = int(kwargs.get('batch_size'))
        self.filepath = kwargs.get('filepath')
        self.tokenizer = kwargs.get('tokenizer')
        self.sequences = kwargs.get('sequences')
        self.MAX_NUM_WORDS = kwargs.get('max_words')
        self.MAX_SEQUENCE_LENGTH = kwargs.get('max_len')
      #  self.word_index = kwargs.get('word_index')


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


    def generate(self):
        fill = 0
        y = []
        x0 = []
        x1 = []
        for line in self.sequences:
            if fill < self.bs:
                if len(line) > 5: # param
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
