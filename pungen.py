from word_predict import WordPredict
from similar import WordSimilarity
from generator import Generator
from retrieve import Retrieve
from dac import DAC

from tensorflow.keras.layers import Embedding
from tensorflow.keras.initializers import Constant
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model

import numpy as np

import nltk
from nltk.tokenize import word_tokenize
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')



import os

PUN_DATA = 'puns.pkl'
TEXT_DATA = 'all.txt'
BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'embs/glove.6B')
PUN_DATA_DIR = os.path.join('', 'data/semeval/')
TEXT_DATA_DIR = os.path.join('', 'data/bookcorpus/')


MIN_SEQ_LEN = 5
TOKEN_FILTER = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n©'


# Corpus Parse
MAX_NUM_WORDS = 300000
MAX_LEN = 50
EMBEDDING_DIM = 50

# Predict Model
PREDICT_SPLIT = 0.15
PREDICT_BS = 128
PREDICT_EPOCHS = 20

# Smoother Model
SMOOTH_SPLIT = 0.15
SMOOTH_BS = 16
SMOOTH_EPOCHS = 10

class Pungen:
    def __init__(self, **kwargs):
        self.filepath = kwargs.get('filepath')
        self.embedding_layer = None

    def _parse_corpus(self, min_seq_len, filepath):
        print('Indexing word vectors.')
        self.texts = []
        with open(filepath, encoding='utf-8') as fp:
            for line in fp:
                if line == "\n":
                    continue
                self.texts.append(line)

        self.tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters=TOKEN_FILTER)
        self.tokenizer.fit_on_texts(self.texts)
        self.sequences = self.tokenizer.texts_to_sequences(self.texts)
        self.sequences = [x for x in self.sequences if len(x) >= min_seq_len]
        self.word_index = self.tokenizer.word_index
        print('Found %s unique tokens.' % len(self.word_index))

        print('Found %s texts.' % len(self.sequences))

    def prepare_emb(self, emb_dim, input_length):
        print('Indexing word vectors.')

        emb_name = 'glove.6B.' + str(emb_dim) + "d.txt"

        self.embeddings_index = {}
        with open(os.path.join(GLOVE_DIR, emb_name), encoding='utf-8') as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, 'f', sep=' ')
                self.embeddings_index[word] = coefs

        print('Found %s word vectors.' % len(self.embeddings_index))
        # prepare embedding matrix
        num_words = MAX_NUM_WORDS
        self.embedding_matrix = np.zeros((num_words, emb_dim))
        for word, i in self.word_index.items():
            if i >= num_words:
                continue
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector

        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        self.embedding_layer = Embedding(num_words,
                                         emb_dim,
                                         embeddings_initializer=Constant(self.embedding_matrix),
                                         input_length=input_length,
                                         trainable=False)

    def check_generator(self):
        texts = self.tokenizer.sequences_to_texts(self.sequences)

        if len(texts) != len(self.texts):
            print("Different sizes of texts")
            return

        filter = set(TOKEN_FILTER)

        for i in range(len(texts)):
            if texts[i].lower() != self.texts[i][:-1].lower():

                if any((c in filter) for c in self.texts[i][:-1].lower()):
                    continue

                print(texts[i], self.texts[i][:-1])
                print(self.texts[i][:-1].lower())
                print("Tokenizer failed to tokenize properly!")
                return

        print("Tokenizer check was succesfull!")

    def form_pun(self):
        retrieve = Retrieve(sentence_path=TEXT_DATA_DIR + TEXT_DATA, pun_path=PUN_DATA_DIR + PUN_DATA)
        (pun, sentence, score) = retrieve.retrieve()

        if not sentence:
            print("No sentence with word {} was found. Exiting...".format(pun[1]))
            exit()

        text = word_tokenize(sentence)
        tokenized = nltk.pos_tag(text)

        index = -1
        topic_word = None
        for (word, pos) in tokenized:
            index = index + 1
            if pos == 'NN':
                topic_word = word
                print(word, pos)
                break

        index = -1
        for (word, pos) in tokenized:
            index = index + 1
            if word == pun[1]:
                print(word, pos)
                break

        print(sentence, pun[0], pun[1])
        pre = self.tokenizer.texts_to_sequences([sentence])
        wp = self.tokenizer.texts_to_sequences([pun[0]])
        wa = self.tokenizer.texts_to_sequences([pun[1]])
        print(pre)
        print(wp, wa)
        pre[index] = wp

        post = self.tokenizer.sequences_to_texts(pre)
        print(post)

        #wordsimilarity = WordSimilarity()
        #wordsimilarity.word2vec()
        #wordsimilarity.load()
        #result = wordsimilarity.getSimilar([pun[0]], 5)
        #print(result)

    def train_predict_model(self, model_params):
        predict_word = WordPredict(max_len=MAX_LEN, max_words=MAX_NUM_WORDS, emb_layer=self.embedding_layer)
        predict_word.build_model(**model_params)
        predict_word.compile_model(model_params)

        generator = Generator(sequences=self.sequences, batch_size=PREDICT_BS,
                          max_words=MAX_NUM_WORDS, max_len=MAX_LEN,
                          split=PREDICT_SPLIT)

        predict_word.train(generator, PREDICT_BS, PREDICT_SPLIT, PREDICT_EPOCHS)
        return predict_word

    def load_predict_model(self, path):
        predict_word = load_model(path)
        return predict_word

    def train_dac_model(self, model_params):
        dac = DAC()
        smoother_model = dac.build_model(hidden_sizes=[64, 64], seq_len=50, no_words=40000,
                                      emb_layer=self.embedding_layer, lr=0.01)
        generator = Generator(sequences=self.sequences, batch_size=SMOOTH_BS,
                                   max_words=MAX_NUM_WORDS, max_len=MAX_LEN,
                                   split=SMOOTH_SPLIT)
        smoother_model = dac.train(generator, full_model=smoother_model, model_params=model_params,
                                   bs=SMOOTH_BS, split=SMOOTH_SPLIT, pretrain_epochs=4,
                                   epochs=SMOOTH_EPOCHS)

    def run(self, predict_path, smoother_path):
        self._parse_corpus(MIN_SEQ_LEN, TEXT_DATA_DIR + TEXT_DATA)
        self.prepare_emb(EMBEDDING_DIM, MAX_LEN)

        predict_model = None
        if predict_path is None:
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
            predict_model = self.train_predict_model(model_params)
        else:
            pass
            #predict_model = self.load_predict_model(predict_path)

        #smoother_model = None
        if smoother_path is None:
            model_params = {
                'size': [64, 64],
                'lr': 0.01
            }
            #smoother_model = self.train_dac_model(model_params)
        else:
            pass

        #GENERATE PUN
        pungen.form_pun()

if __name__ == '__main__':
    pungen = Pungen(filepath='data/bookcorpus/all.txt')
    #'models/smoother/1589835246 - pretraining Epoch 04.hdf5'
    pungen.run('models/1589665956 Epoch 20.hdf5', None)
