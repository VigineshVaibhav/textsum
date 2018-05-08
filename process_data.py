import os
import sys
from functions_textsum import UnPickleData, UnPickleDict, NumpyData, ShiftRight, Np2H5
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical                      # for one-hot encodding
import numpy as np


path_train = './data/News_size:25818_tokens:50_type:TRAIN_DATA.pickle'   # TO CHANGE

path_test = './data/News_size:10545_tokens:50_type:TEST_DATA.pickle'



path_emb_dict = './embedings/ALL_words:139667_type:EMBED_DICTONARY.pickle'

n_words = 40000     # only use top n_words most frequent

def ProcessData(path_h5, contents, titles, word_index, emb_dict, n_in, emb_dim):
    '''------------------NUMPY DATA------------------------------'''
    print '\nNumpy Data....'
    # Create Numpy of input data for index and embeddings
    x_emb, x_index = NumpyData(contents, word_index, emb_dict, n_input=n_in, n_emb=emb_dim)
    print '\tx_emb x_index Done!'


    del contents

    # Create Numpy output data for index and embeddings
    y_emb, y_index = NumpyData(titles, word_index, emb_dict, n_input=n_out, n_emb=emb_dim)
    print '\ty_emb y_index Done!'
    #del titles

    print y_index[0]
    print titles[0]
    print len(y_emb[0])

    print len(titles[0])
    max_index = np.amax(y_index) + 1        # +1 because word index starts from 0
    y_hot = []

    for index_title in y_index:
        encoded = to_categorical(index_title, num_classes=max_index)
        y_hot.append(encoded)
        del encoded

    print y_hot[0][3][:15]
    print len(y_hot[0])
    print 'max ',np.amax(y_index)

    sys.exit()


    print '\t Saving all numpys...'
    Np2H5(path_h5, x_emb, x_index, y_emb, y_index)
    print '\tDone!'
    print 'Numpy Data Finished!\n'

    print x_emb.shape
    print x_index.shape

    print y_emb.shape
    print y_index.shape
    del x_emb, x_index, y_emb, y_index  # clean memory
    print '%s %s Finished Processing!\n'%(media, partition_type)

    return


'''------------------LOAD WORD EMBEDDINGS------------------------------'''
print '\nLoad emb_dict...'
emb_dict = UnPickleDict(path_emb_dict)
print 'Finished\n'



''' PARAMETERS '''
data_size = 0
emb_dim = 300           # embedding size is always 300
n_in = 50               # number of tokens of artice content
n_out = 10


''' TRAIN DATA '''
media = path_train.split('/')[-1].split('_')[0]              # News or Blog
partition_type = path_train.split('/')[-1].split('_')[3].split(':')[1]

print '\n\n------------PROCESSING %s %s DATA--------------\n'%(media, partition_type)

'\nUnPickle data...'
contents, titles = UnPickleData(path_train)     # train contents and titles
print 'UnPickle Finished!\n'

data_size = len(titles)
path_h5 = './h5files/%s_size:%s_tokens:%s_type:%s_NUMPYS.h5'%(media, data_size, n_in, partition_type)

print '\nProcessing Data...'
print 'Creating Tokenizer'
t = Tokenizer(num_words=None)
print 'Fitting tokenizer to text...'
t.fit_on_texts(contents + titles)
print 'Creating word index dictionary...'
word_index = {}
for word, index in (t.word_index).items:
    # Use only top n_words
    if index > n_words:
        continue
    word_index[word]
del t
word_index['<UNK>'] = n_words + 1   # add unknown word index at the end of word index
word_index['<EOS>'] = n_words + 1   # add EOS at the end of words index
word_index[''] = 0                  # add zero to word index used when padding with zero

print 'Vocabulary size: ',len(word_index.items)

print 'Finsihed!\n'

ProcessData(path_h5, contents, titles, word_index, emb_dict, n_in, emb_dim)
del contents, titles


''' TESTING DATA '''
media = path_test.split('/')[-1].split('_')[0]              # News or Blog
partition_type = path_test.split('/')[-1].split('_')[3].split(':')[1]

print '\n\n------------PROCESSING %s %s DATA--------------\n'%(media, partition_type)

print '\nUnPickle data...'
contents, titles = UnPickleData(path_test)     # train contents and titles
print 'UnPickle Finished!\n'

data_size = len(titles)
path_h5 = './h5files/%s_size:%s_tokens:%s_type:%s_NUMPYS.h5'%(media, data_size, n_in, partition_type)

ProcessData(path_h5, contents, titles, word_index, emb_dict, n_in, emb_dim)
del contents, titles




print '\nProgram Finished!'


