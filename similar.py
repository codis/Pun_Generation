import os
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'embs/glove.6B/')

class WordSimilarity:
    """
      built after https://machinelearningmastery.com/develop-word-embeddings-python-gensim/

      Gensim docs
      https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.similar_by_word
    """
    def word2vec(self):
        glove_input_file = GLOVE_DIR + 'glove.6B.50d.txt'
        word2vec_output_file = 'glove.6B.50d.txt.word2vec'
        glove2word2vec(glove_input_file, word2vec_output_file)

    def load(self):
        # load the Stanford GloVe model
        filename = 'glove.6B.50d.txt.word2vec'
        self.model = KeyedVectors.load_word2vec_format(filename, binary=False)

    def getSimilar(self, pos_words, neg_words, topn):
        print("cosine similarity between a simple mean of the projection weight vectors")
        result = self.model.most_similar(positive=pos_words, negative=neg_words, topn=topn)
        print(result)
        return result

    def get_cosmul(self, words, topn):
        print("using the multiplicative combination objective")
        result = self.model.most_similar_cosmul(positive=words, topn=topn)
        print(result)

    def average_similar(self):
        print("self made average")
        v1 = self.model.get_vector("meat")
        v2 = self.model.get_vector("man")
        vec = (v1 + v2) / 2
        words = self.model.similar_by_vector(vec)
        print(words)

    def example(self):
        self.word2vec()
        self.load()
        self.average_similar()
        self.get_cosmul(["man", "meat"], 10)
        result = self.getSimilar(["meat", "man"], [], 10)
        result = self.getSimilar(["meet", "meat"], ["man"], 10)
        result = self.getSimilar(["man"], ["meet", "meat"], 10)
        result = self.getSimilar(["man", "meat"], ["meet"], 10)
        result = self.getSimilar(["meet"], ["man", "meat"], 10)

if __name__ == '__main__':
    wordsimilarity = WordSimilarity()
    wordsimilarity.example()
