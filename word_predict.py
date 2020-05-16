from __future__ import print_function

import os
import sys
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Input, Concatenate, Add
from tensorflow.keras.layers import Embedding, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant
from generator import Generator


BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'embs/glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, 'data/bookcorpus')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2

class WordPredict:
    def __init__(self, **kwargs):
        self.MAX_SEQUENCE_LENGTH = kwargs.get('max_len')
        self.EMBEDDING_DIM = kwargs.get('emb_dim')
        self.MAX_NUM_WORDS = kwargs.get('max_words')
        self.VOCAB_SIZE = kwargs.get('vocab_size')

    def test_dataset(self):
        print('Indexing word vectors.')

        self.embeddings_index = {}
        with open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'), encoding='utf-8') as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, 'f', sep=' ')
                self.embeddings_index[word] = coefs

        print('Found %s word vectors.' % len(self.embeddings_index))
        # second, prepare text samples and their labels
        print('Processing text dataset')

        self.texts = []  # list of text samples
        self.labels_index = {}  # dictionary mapping label name to numeric id
        self.labels = []  # list of label ids
        for name in sorted(os.listdir(TEXT_DATA_DIR)):
            path = os.path.join(TEXT_DATA_DIR, name)
            if os.path.isdir(path):
                self.label_id = len(self.labels_index)
                self.labels_index[name] = self.label_id
                for fname in sorted(os.listdir(path)):
                    fpath = os.path.join(path, fname)
                    args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
                    with open(fpath, **args) as f:
                        t = f.read()
                        i = t.find('\n\n')  # skip header
                        if 0 < i:
                            t = t[i:]
                        self.texts.append(t)
                    self.labels.append(self.label_id)

        print('Found %s texts.' % len(self.texts))

        # finally, vectorize the text samples into a 2D integer tensor
        self.tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
        self.tokenizer.fit_on_texts(self.texts)
        self.sequences = self.tokenizer.texts_to_sequences(self.texts)

        self.word_index = self.tokenizer.word_index
        print('Found %s unique tokens.' % len(self.word_index))

        self.data = pad_sequences(self.sequences, maxlen=MAX_SEQUENCE_LENGTH)

        self.labels = to_categorical(np.asarray(self.labels))
        print('Shape of data tensor:', self.data.shape)
        print('Shape of label tensor:', self.labels.shape)

        # split the data into a training set and a validation set
        self.indices = np.arange(self.data.shape[0])
        np.random.shuffle(self.indices)
        self.data = self.data[self.indices]
        self.labels = self.labels[self.indices]
        self.num_validation_samples = int(VALIDATION_SPLIT * self.data.shape[0])

        self.x_train = self.data[:-self.num_validation_samples]
        self.y_train = self.labels[:-self.num_validation_samples]
        self.x_val = self.data[-self.num_validation_samples:]
        self.y_val = self.labels[-self.num_validation_samples:]

        print('Preparing embedding matrix.')


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

    def build_model(self, **kwargs):
        lstm = kwargs.get('lstm_array', None)
        cnn = kwargs.get('cnn_array', None)
        merge_layer = kwargs.get('merge_layer')
        dense = kwargs.get('dense')

        inp_pre = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32')
        inp_post = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32')

        self.prepare_emb()
        emb_pre = self.embedding_layer(inp_pre)
        emb_post = self.embedding_layer(inp_post)

        l_pre = emb_pre
        l_post = emb_post
        if lstm:
            for layer in lstm:
                l_pre = LSTM(layer['pre']['size'])(l_pre)
                l_post = LSTM(layer['post']['size'])(l_post)

        if merge_layer == 'concat':
            merge = Concatenate()([l_pre, l_post])
        if merge_layer == 'add':
            merge = Add()([l_pre, l_post])

        x_dense = merge
        for layer in dense:
            x_dense = Dense(layer['size'], activation=layer['act'])(x_dense)

        out = Dense(self.VOCAB_SIZE, activation='softmax')(x_dense)

        self.model = Model(inputs=[inp_pre, inp_post], outputs=out)

    def train(self):
        gen = Generator(filepath='all.txt', batch_size=32)
        train_gen = gen.generate()
        test_gen = gen.generate()
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy')

        self.model.fit(train=train_gen, test=test_gen, epochs=5)


if __name__ == '__main__':
    model_params = {
        'lstm':[
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
    model = WordPredict(max_len=1000, emb_dim=300, max_words=20000, vocab_size=40000)
    model.build_model(**model_params)
    model.model.summary()
    model.train()