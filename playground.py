import json
import gzip
from keras.preprocessing.text import text_to_word_sequence
import numpy as np
from gensim.models import KeyedVectors

import cPickle as pickle

from functions_textsum import *
import random


path_data = './data/News_size:25818_tokens:50_type:TRAIN_DATA.pickle'
path_numpy = './h5files/News_size:25818_tokens:50_type:TRAIN_NUMPYS.h5'

''' READING INPUT '''
print '\nReading H5 file to numpys...'
x_emb, x_index, y_emb, y_index = H52Np(path_numpy)
print 'Finished!\n'

print '\nUnPickle data..'
contents, titles = UnPickleData(path_data)
print 'Finsihed!\n'

print len(x_emb[1])
print len(contents[1])

print x_index[1]
print contents[1]
print '\n\nend'
