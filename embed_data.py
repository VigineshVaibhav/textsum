import os
import sys
from gensim.models import KeyedVectors
from functions_textsum import PickleDict
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
import cPickle as pickle
import numpy as np

'''
    Go through all data and create a set of words
    Use Google News word embeddings to get all word embedding form our corpus
'''



path_news= './data/News_size:734488_ALL.pickle'

path_blog = './data/Blog_size:265512_ALL.pickle'

path_google_model = './embedings/GoogleNews-vectors-negative300.bin'

path_emb_dict = './embedings/'

start_flag = '-#,##_-#,##'
end_flag = '##,###.#_##,###.#_##,###.#_##,###.#'

print '\nUnPickle data...'
corpus_set = set()      # set of all words in corpus
print 'News...'
with (open(path_news, "rb")) as openfile:
    while True:
      try:
        _, content, title, _ , _ , _ = pickle.load(openfile)
        corpus_set.update(content)
        corpus_set.update(title)
        del content, title
      except EOFError:
        break
print 'News done!'
print 'Blog...'
with (open(path_blog, "rb")) as openfile:
    while True:
      try:
        _, content, title, _ , _ , _ = pickle.load(openfile)
        corpus_set.update(content)
        corpus_set.update(title)
        del content, title
      except EOFError:
        break
print 'Blog done!'
print 'UnPickle Finished!\n'


print '\nLoad Word2Vec model using Gensim...'
google_model = KeyedVectors.load_word2vec_format(path_google_model, binary=True)
google_dict = {}
vocab = list(google_model.vocab)
for word in vocab:
	#wrd = word.lower()
	google_dict[word] = google_model[word]
del google_model
print 'Finished Loading %s word embeddings!'%(len(google_dict.items()))


print '\nCreating word embedings dictionary...'
emb_dict = {}   # embedding of dictionary
for word in corpus_set:    # look in all corpus (train + test)
    if word in google_dict:
        emb_dict[word] = google_dict[word]


if start_flag in emb_dict:
    print '\nError Start Flag!\n'
    sys.exit()
elif start_flag in google_dict:
    emb_dict['<Start>'] = google_dict[start_flag]
    print '\nStart Flag Added!\n'
else:
    print '\nError Start Flag!\n'
    sys.exit()

if end_flag in emb_dict:
    print '\nError End Flag!\n'
    sys.exit()
elif end_flag in google_dict:
    emb_dict['<End>'] = google_dict[end_flag]
    print '\nEnd Flag Added!\n'
else:
    print '\nError End Flag!\n'
    sys.exit()

del google_dict
print 'Finsihed!\n'


print '\nPickle emb_dict..'
# we have word embeddings of all
path_emb_dict += 'ALL_words:%s_type:EMBED_DICTONARY.pickle'%(len(emb_dict.items()))
PickleDict(path_emb_dict, emb_dict)
del emb_dict
print 'Finished!\n'



print '\nProgram Finished!'
