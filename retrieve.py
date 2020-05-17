import pickle
import random
import sys
import os
import re

class Retrieve:

    def __init__(self, **kwargs):
        self.sentence_path = kwargs.get('sentence_path')
        self.pun_path = kwargs.get('pun_path')

    def get_pun(self):
        with open(self.pun_path, 'rb') as f:
            pair_puns = pickle.load(f)

        return random.choice(pair_puns)

    def value(self, sentence):
        return 1

    def length_score(self, words):
        return 0 - len(words)

    def possition_score(self, index, words):
        last_index = index[-1]

        return last_index

    def eval(self, index, words):
        return self.length_score(words) + self.possition_score(index, words)

    def get_sentence(self, pun):
        (_, wa) = pun
        with open(self.sentence_path) as fp:
            max_score = float('-inf')
            best_sentence = []
            for line in fp:
                index = line.rfind(wa)
                words = re.findall(r'\w+', line)
                index = [i for i, x in enumerate(words) if wa == x]

                if len(index) != 0:
                    score = self.eval(index, words) * self.value(line)

                    if score > max_score:
                        max_score = score
                        best_sentence = line

        return max_score, best_sentence

    def retrieve(self):
        if not os.path.isfile(self.sentence_path):
            print("File path for Sentence data: {} does not exist. Exiting...".format(self.sentence_path))
            sys.exit()

        if not os.path.isfile(self.pun_path):
            print("File path for Puns data: {} does not exist. Exiting...".format(self.pun_path))
            sys.exit()

        pun = self.get_pun()
        score, sentence = self.get_sentence(pun)

        return pun, sentence, score

if __name__ == '__main__':
    sentence_path = "data/bookcorpus/all.txt"
    pun_path = "data/semeval/puns.pkl"
    retrieve = Retrieve(sentence_path=sentence_path, pun_path=pun_path)
    (pun, sentence, score) = retrieve.retrieve()