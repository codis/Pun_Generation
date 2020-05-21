from tensorflow.keras.layers import Dense, Input, Concatenate, Add, Bidirectional
from tensorflow.keras.layers import Dropout, LSTM, Flatten, Attention, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from log_callback import LogCalllback
import tensorflow as tf
import os, time
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class DAC:
    def __init__(self, **kwargs):
        self.MAX_SEQ_LEN = kwargs.get('max_seq_len', 50)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)

    def build_model(self, hidden_sizes, seq_len, no_words, emb_layer, lr):
        encoder_inputs = Input(shape=(seq_len,), name='encoder_inputs')
        encoder_emb = emb_layer(encoder_inputs)

        x = encoder_emb

        for size in hidden_sizes:
            x = LSTM(size, return_sequences=True)(x)

        td = TimeDistributed(Dense(no_words, activation='softmax', name='softmax_layer'))(x)

        model = Model(inputs = encoder_inputs, outputs=td)
        opt = Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        model.summary()
        return model

    def inference(self, phrase):
        x = pad_sequences([phrase], self.MAX_SEQ_LEN)
        return np.argmax(self.model.predict(x), axis=-1)

    def train(self, generator, model_params, full_model, bs , split, pretrain_epochs=2, epochs=10):
        """ Training the model """
        prefix = str(int(time.time()))
        models_folder = os.getcwd() + "/models/smoother/" + prefix + " - pretraining Epoch {epoch:02d}.hdf5"

        with open("logs/training_log.txt", "a+") as log:
            log.write('\n' + prefix + '     ' + str(model_params))
        early_stop = EarlyStopping(monitor='val_categorical_accuracy', patience=2, restore_best_weights=True,
                                   mode='max')
        save_callback = ModelCheckpoint(models_folder, monitor='val_categorical_accuracy', verbose=1,
                                        save_best_only=True, mode='max')
        log_callback = LogCalllback(prefix=prefix, log_path=os.getcwd() + "/logs/dac_train_log.csv")

        train_gen = generator.pretrain_gen('train')
        test_gen = generator.pretrain_gen('test')

        full_model.fit(train_gen, validation_data=test_gen,
                       steps_per_epoch=int(len(generator.sequences) * (1 - split) / bs),
                       validation_steps=int(len(generator.sequences) * (split) / bs),
                       callbacks=[
                           early_stop,
                           save_callback,
                           log_callback
                       ],
                       epochs=pretrain_epochs,
                       )

        train_gen = generator.generate_enc_dec2('train')
        test_gen = generator.generate_enc_dec2('test')
        models_folder = os.getcwd() + "/models/smoother/" + prefix + " - training Epoch {epoch:02d}.hdf5"

        early_stop = EarlyStopping(monitor='val_categorical_accuracy', patience=2, restore_best_weights=True,
                                   mode='max')
        save_callback = ModelCheckpoint(models_folder, monitor='val_categorical_accuracy', verbose=1,
                                        save_best_only=True, mode='max')
        log_callback = LogCalllback(prefix=prefix, log_path=os.getcwd() + "/logs/dac_train_log.csv")

        full_model.fit(train_gen, validation_data=test_gen,
                       steps_per_epoch=int(len(generator.sequences) * (1 - split) / bs),
                       validation_steps=int(len(generator.sequences) * (split) / bs),
                      # steps_per_epoch=int(33752/4),
                      # validation_steps=int(6132/4),
                       callbacks=[
                           early_stop,
                           save_callback,
                           log_callback
                       ],
                       epochs=epochs,
                      )

if __name__ == '__main__':
    dac = DAC()
    dac.load_model('models/smoother/1589834963 - training Epoch 09.hdf5')
    smoothed = dac.inference( [0, 3990, 0, 686, 2569, 8, 547, 44, 1, 1472])
# [9, 3990, 19, 686, 2569, 8, 547, 44, 1, 1472]
    print(smoothed)