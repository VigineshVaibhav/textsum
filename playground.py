import json
import gzip
from keras.preprocessing.text import text_to_word_sequence
import numpy as np
from gensim.models import KeyedVectors

import cPickle as pickle

from functions_textsum import *
import random



path_google_model = './embedings/GoogleNews-vectors-negative300.bin'

def FlipNumpy(np_in):
    # flip numpy on axis 1
    np_flip = np.flip(np_in, 1)
    np_out = np.zeros(np_in.shape)
    for i_l, line in enumerate(np_flip):
        offset = 0
        for i_v, value in enumerate(line):
            if value == 0:
                offset += 1
                continue
            else:
                np_out[i_l][i_v - offset] = value
    del np_flip
    return np_out

a = np.ones(300)
b = np.random.random(300)
start = '-#,##_-#,##'
end = '##,###.#_##,###.#_##,###.#_##,###.#'

print '\nLoad Word2Vec model using Gensim...'
google_model = KeyedVectors.load_word2vec_format(path_google_model, binary=True)
google_dict = {}
vocab = list(google_model.vocab)
for word in vocab:
	#wrd = word.lower()
	google_dict[word] = google_model[word]
del google_model
print 'Finished Loading %s word embeddings!'%(len(google_dict.items()))


emb_dict = UnPickleDict('./embedings/ALL_words:139665_type:EMBED_DICTONARY.pickle')
m = float('-inf')
w = ''
e = None
for word, emb in emb_dict.items():
    if word == start:
            print word
            print emb
    if word == end:
        print emb
        print word

print 'end'
