from __future__ import print_function

import os
import sys
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant


BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'embs/glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, 'data/20_newsgroup')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector


class ExampleWordClassif:
    """
    built after https://keras.io/examples/pretrained_word_embeddings/
    """
    def load_embs(self):
        print('Indexing word vectors.')

        self.embeddings_index = {}
        with open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'), encoding='utf-8') as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, 'f', sep=' ')
                self.embeddings_index[word] = coefs

        print('Found %s word vectors.' % len(self.embeddings_index))

    def load_data(self):
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
                    if fname.isdigit():
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

    def tokenize(self):
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
        # prepare embedding matrix
        self.num_words = min(MAX_NUM_WORDS, len(self.word_index) + 1)
        self.embedding_matrix = np.zeros((self.num_words, EMBEDDING_DIM))
        for word, i in self.word_index.items():
            if i >= MAX_NUM_WORDS:
                continue
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector

        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        self.embedding_layer = Embedding(self.num_words,
                                    EMBEDDING_DIM,
                                    embeddings_initializer=Constant(self.embedding_matrix),
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)

    def train(self):
        print('Training model.')

        # train a 1D convnet with global maxpooling
        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = self.embedding_layer(sequence_input)
        x = Conv1D(128, 5, activation='relu')(embedded_sequences)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(128, activation='relu')(x)
        preds = Dense(len(self.labels_index), activation='softmax')(x)

        model = Model(sequence_input, preds)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])

        model.fit(self.x_train, self.y_train,
                  batch_size=128,
                  epochs=10,
                  validation_data=(self.x_val, self.y_val))

    def example(self):
        self.load_embs()
        self.load_data()
        self.tokenize()
        self.prepare_emb()

if __name__ == '__main__':
    wordpredict = ExampleWordClassif()
    wordpredict.example()
