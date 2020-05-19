from __future__ import print_function

import os
import time

from tensorflow.keras.optimizers import Adam, Adagrad, RMSprop, SGD
from tensorflow.keras.layers import Dense, Input, Concatenate, Add
from tensorflow.keras.layers import Dropout, LSTM, Flatten
from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from log_callback import LogCalllback


class WordPredict:
    def __init__(self, **kwargs):
        self.MAX_SEQUENCE_LENGTH = kwargs.get('max_len')
        self.MAX_NUM_WORDS = kwargs.get('max_words')
        self.embedding_layer = kwargs.get('emb_layer')

    def build_model(self, **kwargs):
        lstm = kwargs.get('lstm_array', None)
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
                l_pre = LSTM(layer)(l_pre)
                l_post = LSTM(layer)(l_post)

        if merge_layer == 'concat':
            merge = Concatenate()([l_pre, l_post])
        if merge_layer == 'add':
            merge = Add()([l_pre, l_post])

        x_dense = Flatten()(merge)
        for size in dense['size']:
            x_dense = Dense(size, activation=dense['act'])(x_dense)
            x_dense = Dropout(dense['dropout'])(x_dense)
        out = Dense(self.MAX_NUM_WORDS, activation='softmax')(x_dense)

        self.model = Model(inputs=[inp_pre, inp_post], outputs=out)

    def compile_model(self, model_params):
        self.model_params = model_params
        if model_params['optimizer'] == 'adam':
            optimizer = Adam(learning_rate=model_params['lr'])
        elif model_params['optimizer'] == 'adagrad':
            optimizer = Adagrad(learning_rate=model_params['lr'])
        elif model_params['optimizer'] == 'rmsprop':
            optimizer = RMSprop(learning_rate=model_params['lr'])
        elif model_params['optimizer'] == 'sgd':
            optimizer = SGD(learning_rate=model_params['lr'])

        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['categorical_accuracy'])


    def train(self, generator, bs, split, epochs=5):
        train_gen = generator.generate(dataset='train')
        test_gen = generator.generate(dataset='test')

        prefix = str(int(time.time()))
        models_folder = os.getcwd() + "/models/" + prefix + " Epoch {epoch:02d}.hdf5"
        with open("logs/training_log.txt", "a+") as log:
            log.write('\n' + prefix + '     ' + str(self.model_params))

        early_stop = EarlyStopping(monitor='val_categorical_accuracy', patience=2, restore_best_weights=True,
                                   mode='max')
        save_callback = ModelCheckpoint(models_folder, monitor='val_categorical_accuracy', verbose=1,
                                        save_best_only=True, mode='max')
        log_callback = LogCalllback(prefix=prefix, log_path=os.getcwd() + "/logs/train_log.csv")
        self.model.fit(train_gen, validation_data=test_gen,
                       steps_per_epoch=int(len(generator.sequences) * (1 - split) / bs),
                       validation_steps=int(len(generator.sequences) * (split) / bs),
                       epochs=epochs,
                       callbacks=[
                           early_stop,
                           save_callback,
                           log_callback
                       ])

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