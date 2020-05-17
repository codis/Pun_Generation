from word_predict import WordPredict
from similar import WordSimilarity
from generator import Generator
from retrieve import Retrieve

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam, Adagrad, RMSprop, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from log_callback import LogCalllback
import time
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')



import os
import sys

BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'embs/glove.6B')

class Pungen:
    def __init__(self, **kwargs):
        self.MAX_SEQUENCE_LENGTH = kwargs.get('max_len')
        self.EMBEDDING_DIM = kwargs.get('emb_dim')
        self.MAX_NUM_WORDS = kwargs.get('max_words')
        self.TEXT_DATA_DIR = os.path.join('', 'data/bookcorpus/')
        self.PUN_DATA_DIR = os.path.join('', 'data/semeval/')
        self.bs = int(kwargs.get('batch_size'))
        self.filepath = kwargs.get('filepath')
        self.split = kwargs.get('split')
        #self._parse_corpus(min_seq_len=5)
        #self.prepare_emb()

    def create_model(self, model_params):
        word_predict = WordPredict(emb_layer=self.embedding_layer,
                                   max_len=self.MAX_SEQUENCE_LENGTH, emb_dim=self.EMBEDDING_DIM,
                                   max_words=self.MAX_NUM_WORDS)
        word_predict.build_model(**model_params)
        self.model = word_predict.model
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

    def train(self, epochs=5):
        gen = Generator(filepath='all.txt', batch_size=self.bs,
                        tokenizer=self.tokenizer, sequences=self.sequences,
                        max_words=self.MAX_NUM_WORDS, max_len=self.MAX_SEQUENCE_LENGTH,
                        split=self.split)
        train_gen = gen.generate(dataset='train')
        test_gen = gen.generate(dataset='test')
        prefix = str(int(time.time()))
        models_folder = os.getcwd() + "/models/" + prefix + " Epoch {epoch:02d}.hdf5"

        with open("logs/training_log.txt", "a+") as log:
            log.write('\n' + prefix + '     ' + str(self.model_params))

        early_stop = EarlyStopping(monitor='val_categorical_accuracy', patience=2, restore_best_weights=True, mode='max')
        save_callback = ModelCheckpoint(models_folder, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
        log_callback = LogCalllback(prefix=prefix, log_path = os.getcwd() + "/logs/train_log.csv")
        self.model.fit(train_gen, validation_data=test_gen,
                       steps_per_epoch= int(len(self.sequences) * (1-self.split) / self.bs),
                       validation_steps=int(len(self.sequences) * (self.split) / self.bs),
                       epochs=epochs,
                       callbacks=[
                           early_stop,
                           save_callback,
                           log_callback
                       ])

    def prepare_emb(self):
        print('Indexing word vectors.')

        self.embeddings_index = {}
        if self.EMBEDDING_DIM == 50:
            emb_name = 'glove.6B.50d.txt'
        if self.EMBEDDING_DIM == 100:
            emb_name = 'glove.6B.100d.txt'
        if self.EMBEDDING_DIM == 200:
            emb_name = 'glove.6B.200d.txt'
        if self.EMBEDDING_DIM == 300:
            emb_name = 'glove.6B.300d.txt'

        with open(os.path.join(GLOVE_DIR, emb_name), encoding='utf-8') as f:
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

    def _parse_corpus(self, min_seq_len):
        print('Indexing word vectors.')
        self.texts = []
        with open(self.filepath, encoding='utf-8') as fp:
            for line in fp:
                if line == "\n":
                    continue
                self.texts.append(line)

        self.filter = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n©'
        self.tokenizer = Tokenizer(num_words=self.MAX_NUM_WORDS, filters=self.filter)  # params
        self.tokenizer.fit_on_texts(self.texts)
        self.sequences = self.tokenizer.texts_to_sequences(self.texts)
        self.sequences = [x for x in self.sequences if len(x) >= min_seq_len]
        self.word_index = self.tokenizer.word_index
        print('Found %s unique tokens.' % len(self.word_index))

        print('Found %s texts.' % len(self.sequences))

    def form_pun(self):
        retrieve = Retrieve(sentence_path=self.TEXT_DATA_DIR + "all.txt", pun_path=self.PUN_DATA_DIR + "puns.pkl")
        (pun, sentence, score) = retrieve.retrieve()

        if not sentence:
            print("No sentence with word {} was found. Exiting...".format(pun[1]))
            exit()

        text = word_tokenize(sentence)
        tokenized = nltk.pos_tag(text)

        print(tokenized)
        print(pun, sentence, score)

        index = -1
        topic_word = None
        for (word, pos) in tokenized:
            index = index + 1
            if pos == 'NN':
                topic_word = word
                print(word, pos)
                break

        wordsimilarity = WordSimilarity()
        wordsimilarity.word2vec()
        wordsimilarity.load()
        result = wordsimilarity.getSimilar([topic_word, pun[1]], 5)
        print(result)



    def check_generator(self):
        texts = self.tokenizer.sequences_to_texts(self.sequences)

        if len(texts) != len(self.texts):
            print("Different sizes of texts")
            return

        filter = set(self.filter)

        for i in range(len(texts)):
            if texts[i].lower() != self.texts[i][:-1].lower():

                if any((c in filter) for c in self.texts[i][:-1].lower()):
                    continue

                print(texts[i], self.texts[i][:-1])
                print(self.texts[i][:-1].lower())
                print("Tokenizer failed to tokenize properly!")
                return

        print("Tokenizer check was succesfull!")
#{'lstm': [16], 'dense': {'size': [64, 32], 'dropout': 0, 'act': 'elu'}, 'merge_layer': 'concat', 'optimizer': 'adam', 'lr': 0.0005}
if __name__ == '__main__':
    model_params = {
        'lstm': [16],
        'merge_layer': 'concat',
        'dense':
            {
                'size': [64, 32],
                'act': 'elu',
                'dropout': 0
            },
        'optimizer': 'adam',
        'lr': 0.0005
    }



    pungen = Pungen(filepath='data/news_aggregated/all.txt', batch_size=128, max_len=50,
               emb_dim=50, max_words=300000, split=0.15)
    pungen.form_pun()
    #pungen.create_model(model_params=model_params)
    #pungen.train()
    #pungen.check_generator()