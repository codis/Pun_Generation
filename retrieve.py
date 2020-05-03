import sys
import os
import re

def value(sentence):
    return 1

def length_score(words):
    return 0 - len(words)

def possition_score(index, words):
    last_index = index[-1]

    return last_index

def eval(index, words):
    return length_score(words) + possition_score(index, words)

def main():
    filepath = sys.argv[1]
    word = sys.argv[2]

    if not os.path.isfile(filepath):
        print("File path {} does not exist. Exiting...".format(filepath))
        sys.exit()

    with open(filepath) as fp:
        max_score = float('-inf')
        best_sentence = []
        for line in fp:
            index = line.rfind(word)
            words = re.findall(r'\w+', line)
            index = [i for i, x in enumerate(words) if word == x]

            if len(index) != 0:
                score = eval(index, words) * value(line)

                if score > max_score:
                    max_score = score
                    best_sentence = line

        return max_score, best_sentence

if __name__ == '__main__':
    print(main())