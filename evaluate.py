import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
class Evaluate:
    def __init__(self):
        self.model = None
        self.MAX_SEQUENCE_LENGTH = 50
    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)

    def compute_surpisal(self, sentence, pun_word, pun_alternative, context_window):
        pun_index = sentence.index(pun_alternative)
        seq_pre = pad_sequences([sentence[:pun_index]], maxlen=self.MAX_SEQUENCE_LENGTH)
        seq_post = pad_sequences([sentence[pun_index+1:]], maxlen=self.MAX_SEQUENCE_LENGTH)
        probabilities = self.model.predict([seq_pre, seq_post])[0]
        global_prob = probabilities[pun_word]
        global_prob_alt = probabilities[pun_alternative]


        seq_pre_local = pad_sequences([sentence[pun_index - context_window:pun_index]], maxlen=self.MAX_SEQUENCE_LENGTH)
        seq_post_local = pad_sequences([sentence[pun_index+1:pun_index+context_window+1]], maxlen=self.MAX_SEQUENCE_LENGTH)
        local_probabilities = self.model.predict([seq_pre_local, seq_post_local])[0]
        local_prob = local_probabilities[pun_word]
        local_prob_alt = local_probabilities[pun_alternative]

        s_global = - np.log(global_prob_alt/global_prob)
        s_local = - np.log(local_prob_alt/local_prob)
        print("global surprizal - {}".format(s_global))
        print("local surprizal - {}".format(s_local))

        return -np.log(s_local/s_global)

if __name__ == '__main__':
    eval = Evaluate()
    eval.load_model('models/1589672236 Epoch 16.hdf5')
    sentence = [9, 3990, 19, 686, 2569, 8, 547, 44, 1, 1472]
    pun_word = 1472
    pun_alternative = 1829
    context_window = 2
    surprizal = eval.compute_surpisal(sentence=sentence, pun_word=pun_word,
                                      pun_alternative=pun_alternative, context_window=context_window)
    print(surprizal)

