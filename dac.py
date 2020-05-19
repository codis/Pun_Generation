from tensorflow.keras.layers import Dense, Input, Concatenate, Add, Bidirectional
from tensorflow.keras.layers import Dropout, LSTM, Flatten, Attention, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from log_callback import LogCalllback
import os, time
import tensorflow as tf
from pungen import Pungen
import numpy as np
from generator import Generator

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class DAC:
    def __init__(self):
        self.pungen = Pungen(filepath='all.txt', batch_size=16, max_len=50,
                        emb_dim=50, max_words=40000, split=0.15)
    def build_model(self, hidden_size, seq_len, no_words, emb_layer):
        """ Defining a NMT model """

        # Define an input sequence and process it.

        encoder_inputs = Input(shape=(seq_len, ), name='encoder_inputs')
        encoder_emb = emb_layer(encoder_inputs)
        decoder_inputs = Input(shape=(seq_len,), name='decoder_inputs')
        decoder_emb = emb_layer(decoder_inputs)

        # Encoder GRU
        encoder_gru = LSTM(hidden_size, return_sequences=True, return_state=True, name='encoder_gru')
        encoder_out, enc_h, enc_c = encoder_gru(encoder_emb)

        # Set up the decoder GRU, using `encoder_states` as initial state.
        decoder_gru = LSTM(hidden_size, return_sequences=True, return_state=True, name='decoder_gru')
        decoder_out, dec_h, dec_c = decoder_gru([decoder_emb,enc_h, enc_c])

        # Attention layer
        attn_layer = Attention(32,name='attention_layer')
        attn_out = attn_layer([decoder_out,encoder_out])

        # Concat attention input and decoder GRU output
        decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_out, attn_out])

        # Dense layer
        dense = Dense(no_words, activation='softmax', name='softmax_layer')
        dense_time = TimeDistributed(dense, name='time_distributed_layer')
        decoder_pred = dense_time(decoder_concat_input)

        # Full model
        full_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
        full_model.compile(optimizer='adam', loss='categorical_crossentropy')

        full_model.summary()

        return full_model

    def build_model3(self, hidden_sizes, seq_len, no_words, emb_layer, lr):
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

    def build_model2(self, hidden_size, seq_len, no_words, emb_layer, lr):
        """ Defining a NMT model """

        # Define an input sequence and process it.

        encoder_inputs = Input(shape=(seq_len, ), name='encoder_inputs')
        encoder_emb = emb_layer(encoder_inputs)
        decoder_inputs = Input(shape=(seq_len,), name='decoder_inputs')
        decoder_emb = emb_layer(decoder_inputs)

        # Encoder GRU
        encoder_gru = LSTM(hidden_size, return_state=True, name='encoder_gru')
        encoder_out, enc_h, enc_c = encoder_gru(encoder_emb)

        # Set up the decoder GRU, using `encoder_states` as initial state.
        decoder_gru = LSTM(hidden_size, name='decoder_gru')
        decoder_out = decoder_gru([decoder_emb,enc_h, enc_c])

        # Attention layer
        attn_layer = Attention(32,name='attention_layer')
        attn_out = attn_layer([decoder_out,encoder_out])

        # Concat attention input and decoder GRU output
        decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_out, attn_out])

        # Dense layer
        dense = Dense(no_words, activation='softmax', name='softmax_layer')
        decoder_pred = dense(decoder_concat_input)

        # Full model
        full_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
        opt = Adam(lr=lr)
        full_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        full_model.summary()

        return full_model


    def train(self, model_params, full_model,pretrain_epochs=2, epochs=10):
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

        gen = Generator(filepath='all.txt', batch_size=self.pungen.bs,
                        tokenizer=self.pungen.tokenizer, sequences=self.pungen.sequences,
                        max_words=self.pungen.MAX_NUM_WORDS, max_len=self.pungen.MAX_SEQUENCE_LENGTH,
                        split=self.pungen.split)

        train_gen = gen.pretrain_gen('train')
        test_gen = gen.pretrain_gen('test')

        full_model.fit(train_gen, validation_data=test_gen,
                       steps_per_epoch=int(len(self.pungen.sequences) * (1 - self.pungen.split) / self.pungen.bs),
                       validation_steps=int(len(self.pungen.sequences) * (self.pungen.split) / self.pungen.bs),
                       callbacks=[
                           early_stop,
                           save_callback,
                           log_callback
                       ],
                       epochs=pretrain_epochs,
                       )

        train_gen = gen.generate_enc_dec2('train')
        test_gen = gen.generate_enc_dec2('test')
        models_folder = os.getcwd() + "/models/smoother/" + prefix + " - training Epoch {epoch:02d}.hdf5"

        early_stop = EarlyStopping(monitor='val_categorical_accuracy', patience=2, restore_best_weights=True,
                                   mode='max')
        save_callback = ModelCheckpoint(models_folder, monitor='val_categorical_accuracy', verbose=1,
                                        save_best_only=True, mode='max')
        log_callback = LogCalllback(prefix=prefix, log_path=os.getcwd() + "/logs/dac_train_log.csv")

        full_model.fit(train_gen, validation_data=test_gen,
                       steps_per_epoch=int(len(self.pungen.sequences) * (1 - self.pungen.split) / self.pungen.bs),
                       validation_steps=int(len(self.pungen.sequences) * (self.pungen.split) / self.pungen.bs),
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

    model_params = {
        'size':[64, 64],
        'lr': 0.01
    }

    full_model = dac.build_model3(hidden_sizes=[64, 64], seq_len=50, no_words=40000,emb_layer=dac.pungen.embedding_layer, lr=0.01)
    dac.train(full_model=full_model, model_params=model_params, pretrain_epochs=4, epochs=10)
    model_params = {
        'size':[64, 64],
        'lr': 0.01
    }
    full_model = dac.build_model3(hidden_sizes=[128, 64], seq_len=50, no_words=40000,emb_layer=dac.pungen.embedding_layer, lr=0.01)
    dac.train(full_model=full_model, model_params=model_params, pretrain_epochs=4, epochs=10)

    model_params = {
        'size':[128, 64],
        'lr': 0.01
    }
    full_model = dac.build_model3(hidden_sizes=[128, 128], seq_len=50, no_words=40000,emb_layer=dac.pungen.embedding_layer, lr=0.01)
    dac.train(full_model=full_model, model_params=model_params, pretrain_epochs=4, epochs=10)

    model_params = {
        'size':[128, 128],
        'lr': 0.01
    }
    full_model = dac.build_model3(hidden_sizes=[32, 32], seq_len=50, no_words=40000,emb_layer=dac.pungen.embedding_layer, lr=0.01)
    dac.train(full_model=full_model, model_params=model_params, pretrain_epochs=4, epochs=10)

    model_params = {
        'size':[64, 64],
        'lr': 0.05
    }
    full_model = dac.build_model3(hidden_sizes=[64, 64], seq_len=50, no_words=40000,
                                  emb_layer=dac.pungen.embedding_layer, lr=0.05)
    dac.train(full_model=full_model, model_params=model_params, pretrain_epochs=4, epochs=10)

    model_params = {
        'size':[128, 64],
        'lr': 0.05
    }
    full_model = dac.build_model3(hidden_sizes=[128, 64], seq_len=50, no_words=40000,
                                  emb_layer=dac.pungen.embedding_layer, lr=0.05)
    dac.train(full_model=full_model, model_params=model_params, pretrain_epochs=4, epochs=10)

    model_params = {
        'size':[128, 128],
        'lr': 0.05
    }
    full_model = dac.build_model3(hidden_sizes=[128, 128], seq_len=50, no_words=40000,
                                  emb_layer=dac.pungen.embedding_layer, lr=0.05)
    dac.train(full_model=full_model, model_params=model_params, pretrain_epochs=4, epochs=10)

    model_params = {
        'size':[32, 32],
        'lr': 0.05
    }
    full_model = dac.build_model3(hidden_sizes=[32, 32], seq_len=50, no_words=40000,
                                  emb_layer=dac.pungen.embedding_layer, lr=0.05)
    dac.train(full_model=full_model, model_params=model_params, pretrain_epochs=4, epochs=10)


