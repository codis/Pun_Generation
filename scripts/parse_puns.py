import json
import pickle

if __name__ == '__main__':
    filepath = 'data/semeval/'

    pair_puns = []

    with open(filepath + 'dev.json', 'r') as f:
        puns_dict = json.load(f)

    for item in puns_dict:
        pair_puns.append((item['pun_word'], item['alter_word']))

    with open(filepath + 'test.json', 'r') as f:
        puns_dict = json.load(f)

    for item in puns_dict:
        pair_puns.append((item['pun_word'], item['alter_word']))

    with open('data/semeval/puns.pkl', 'wb') as f:
        pickle.dump(pair_puns, f)
