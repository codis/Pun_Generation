import os
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'embs/glove.6B/')

class WordSimilarity:
    """
      built after https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
    """
    def word2vec(self):
        glove_input_file = GLOVE_DIR + 'glove.6B.50d.txt'
        word2vec_output_file = 'glove.6B.50d.txt.word2vec'
        glove2word2vec(glove_input_file, word2vec_output_file)

    def load(self):
        # load the Stanford GloVe model
        filename = 'glove.6B.50d.txt.word2vec'
        self.model = KeyedVectors.load_word2vec_format(filename, binary=False)

    def getSimilar(self, words, topn):
        result = self.model.most_similar(positive=words, topn=topn)
        return result

    def example(self):
        self.word2vec()
        self.load()
        result = self.getSimilar("programmer", 5)
        print(result)

if __name__ == '__main__':
    wordsimilarity = WordSimilarity()
    wordsimilarity.example()
