import pickle
import random
import sys
import os
import re

def get_pun(pun_path):
    with open(pun_path, 'rb') as f:
        pair_puns = pickle.load(f)

    return random.choice(pair_puns)

def value(sentence):
    return 1

def length_score(words):
    return 0 - len(words)

def possition_score(index, words):
    last_index = index[-1]

    return last_index

def eval(index, words):
    return length_score(words) + possition_score(index, words)

def get_sentence(sentence_path, pun):
    (_, wa) = pun
    with open(sentence_path) as fp:
        max_score = float('-inf')
        best_sentence = []
        for line in fp:
            index = line.rfind(wa)
            words = re.findall(r'\w+', line)
            index = [i for i, x in enumerate(words) if wa == x]

            if len(index) != 0:
                score = eval(index, words) * value(line)

                if score > max_score:
                    max_score = score
                    best_sentence = line

    return max_score, best_sentence

def main():
    sentence_path = sys.argv[1]
    pun_path = sys.argv[2]

    if not os.path.isfile(sentence_path):
        print("File path for Sentence data: {} does not exist. Exiting...".format(sentence_path))
        sys.exit()

    if not os.path.isfile(pun_path):
        print("File path for Puns data: {} does not exist. Exiting...".format(pun_path))
        sys.exit()

    pun = get_pun(pun_path)
    print(pun)

    score, sentence = get_sentence(sentence_path, pun)
    print(score, sentence)

if __name__ == '__main__':
    print(main())