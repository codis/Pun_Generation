import random
import sys
import os

def form_sentence_input(sentence):
    words = sentence.split()
    index = random.randint(0, len(words) - 1)

    no_words = len(words)
    to_predict = words[index]
    pre_words = ""
    for i in range(0, index):
        pre_words = pre_words + " " + words[i]
    post_words = ""
    for i in range(index + 1, len(words)):
        post_words = post_words + " " + words[i]

    return (pre_words, post_words, to_predict, no_words)

def form_input(sentences, bs=0):


    return form_sentence_input(sentences)

def main():
    filepath = sys.argv[1]
    bs = sys.argv[2]

    bs = int(bs)

    if not os.path.isfile(filepath):
        print("File path {} does not exist. Exiting...".format(filepath))
        sys.exit()

    with open(filepath) as fp:

        fill = 0
        x = []
        y = []
        for line in fp:

            if line == "\n":
                continue

            if fill < bs:
                (pre, post, to_predict, _) = form_sentence_input(line)
                xi = []
                xi.append(pre)
                xi.append(post)

                x.append(xi)
                y.append(to_predict)
                fill = fill + 1
            else:
                print(x, y)
                fill = 0

if __name__ == '__main__':
    main()