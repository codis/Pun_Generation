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
        self.TEXT_DATA_DIR = os.path.join('', 'data/bookcorpus')

        self.embedding_layer = kwargs.get('emb_layer')


    def build_model(self, **kwargs):
        lstm = kwargs.get('lstm_array', None)
        cnn = kwargs.get('cnn_array', None)
        merge_layer = kwargs.get('merge_layer')
        dense = kwargs.get('dense')

        inp_pre = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32')
        inp_post = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32')

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