from tensorflow.keras.layers import Dense, Input, Concatenate, Add, Bidirectional
from tensorflow.keras.layers import Dropout, LSTM, Flatten, Attention, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from log_callback import LogCalllback
import os, time

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class DAC:
    def __init__(self):
        pass

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

    model_params = {
        'size':[64, 64],
        'lr': 0.01
    }

    full_model = dac.build_model(hidden_sizes=[64, 64], seq_len=50, no_words=40000,emb_layer=dac.pungen.embedding_layer, lr=0.01)
    dac.train(full_model=full_model, model_params=model_params, pretrain_epochs=4, epochs=10)
    model_params = {
        'size':[64, 64],
        'lr': 0.01
    }
    full_model = dac.build_model(hidden_sizes=[128, 64], seq_len=50, no_words=40000,emb_layer=dac.pungen.embedding_layer, lr=0.01)
    dac.train(full_model=full_model, model_params=model_params, pretrain_epochs=4, epochs=10)

    model_params = {
        'size':[128, 64],
        'lr': 0.01
    }
    full_model = dac.build_model(hidden_sizes=[128, 128], seq_len=50, no_words=40000,emb_layer=dac.pungen.embedding_layer, lr=0.01)
    dac.train(full_model=full_model, model_params=model_params, pretrain_epochs=4, epochs=10)

    model_params = {
        'size':[128, 128],
        'lr': 0.01
    }
    full_model = dac.build_model(hidden_sizes=[32, 32], seq_len=50, no_words=40000,emb_layer=dac.pungen.embedding_layer, lr=0.01)
    dac.train(full_model=full_model, model_params=model_params, pretrain_epochs=4, epochs=10)

    model_params = {
        'size':[64, 64],
        'lr': 0.05
    }
    full_model = dac.build_model(hidden_sizes=[64, 64], seq_len=50, no_words=40000,
                                  emb_layer=dac.pungen.embedding_layer, lr=0.05)
    dac.train(full_model=full_model, model_params=model_params, pretrain_epochs=4, epochs=10)

    model_params = {
        'size':[128, 64],
        'lr': 0.05
    }
    full_model = dac.build_model(hidden_sizes=[128, 64], seq_len=50, no_words=40000,
                                  emb_layer=dac.pungen.embedding_layer, lr=0.05)
    dac.train(full_model=full_model, model_params=model_params, pretrain_epochs=4, epochs=10)

    model_params = {
        'size':[128, 128],
        'lr': 0.05
    }
    full_model = dac.build_model(hidden_sizes=[128, 128], seq_len=50, no_words=40000,
                                  emb_layer=dac.pungen.embedding_layer, lr=0.05)
    dac.train(full_model=full_model, model_params=model_params, pretrain_epochs=4, epochs=10)

    model_params = {
        'size':[32, 32],
        'lr': 0.05
    }
    full_model = dac.build_model(hidden_sizes=[32, 32], seq_len=50, no_words=40000,
                                  emb_layer=dac.pungen.embedding_layer, lr=0.05)
    dac.train(full_model=full_model, model_params=model_params, pretrain_epochs=4, epochs=10)


