from word_predict import WordPredict
from generator import Generator
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding
from tensorflow.keras.initializers import Constant

import numpy as np

import os
import sys

BASE_DIR = ''

GLOVE_DIR = os.path.join(BASE_DIR, 'embs/glove.6B')

class Pungen:
    def __init__(self, **kwargs):
        self.MAX_SEQUENCE_LENGTH = kwargs.get('max_len')
        self.EMBEDDING_DIM = kwargs.get('emb_dim')
        self.MAX_NUM_WORDS = kwargs.get('max_words')
        self.VOCAB_SIZE = kwargs.get('vocab_size')
        self.TEXT_DATA_DIR = os.path.join('', 'data/bookcorpus')
        self.bs = int(kwargs.get('batch_size'))
        self.filepath = kwargs.get('filepath')
        self._parse_corpus()
        self.prepare_emb()

    def create_model(self, model_params):
        word_predict = WordPredict(emb_layer=self.embedding_layer,
                                   max_len=1000, emb_dim=300,
                                   max_words=20000, vocab_size=40000)
        word_predict.build_model(**model_params)
        self.model = word_predict.model

    def train(self):
        gen = Generator(filepath='all.txt', batch_size=32, tokenizer='self.tokenizer', sequences=self.sequences)
        train_gen = gen.generate()
        test_gen = gen.generate()
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy')

        self.model.fit(train_gen, validation_data=test_gen, epochs=5)

    def prepare_emb(self):
        print('Indexing word vectors.')

        self.embeddings_index = {}
        with open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'), encoding='utf-8') as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, 'f', sep=' ')
                self.embeddings_index[word] = coefs

        print('Found %s word vectors.' % len(self.embeddings_index))
        # prepare embedding matrix
        self.num_words = self.MAX_NUM_WORDS
        self.embedding_matrix = np.zeros((self.num_words, self.EMBEDDING_DIM))
        for word, i in self.word_index.items():
            if i >= self.MAX_NUM_WORDS:
                continue
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector

        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        self.embedding_layer = Embedding(self.num_words,
                                         self.EMBEDDING_DIM,
                                         embeddings_initializer=Constant(self.embedding_matrix),
                                         input_length=self.MAX_SEQUENCE_LENGTH,
                                         trainable=False)

    def _parse_corpus(self):
        print('Indexing word vectors.')
        self.texts = []
        with open(self.filepath, encoding='utf-8') as fp:
            for line in fp:
                if line == "\n":
                    continue
                self.texts.append(line)

        self.tokenizer = Tokenizer(num_words=20000)  # params
        self.tokenizer.fit_on_texts(self.texts)
        self.sequences = self.tokenizer.texts_to_sequences(self.texts)
        self.word_index = self.tokenizer.word_index
        print('Found %s unique tokens.' % len(self.word_index))

        print('Found %s texts.' % len(self.texts))

if __name__ == '__main__':
    model_params = {
        'lstm': [
            {
                'size': 32
            },
            {
                'size': 16
            }
        ],
        'merge_layer': 'concat',
        'dense': [
            {
                'size': 32,
                'act': 'elu'
            }
        ]
    }

    pungen = Pungen(filepath='all.txt', batch_size=32, max_len=1000,
               emb_dim=300, max_words=20000, vocab_size=40000)
    pungen.create_model(model_params=model_params)
    pungen.train()