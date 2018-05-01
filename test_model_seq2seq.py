#imports
from functions_textsum import *

from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense

from keras.models import load_model
import numpy as np


''' PATH OF NEW MODEL '''
path_model = 	'./models/News_size:None_tokens:50_unit:256_flip:YES_epoch:5_opt:RMSProp_loss:MeanSqarErr_nenc:1_ndec:1_FINALMODEL.h5'

path_data = './data/News_size:10545_tokens:50_type:TEST_DATA.pickle'

path_emb_dict = './embedings/ALL_words:139667_type:EMBED_DICTONARY.pickle'

path_h5 = './h5files/News_size:10545_tokens:50_type:TEST_NUMPYS.h5'

'''DEFINE PARAMETERS '''
media =	path_model.split('/')[-1].split('_')[0]
n_in = int(path_model.split('/')[-1].split('_')[2].split(':')[1])
n_units = int(path_model.split('/')[-1].split('_')[3].split(':')[1])
n_enc_cell = int(path_model.split('/')[-1].split('_')[8].split(':')[1]) - 1
n_dec_cell = int(path_model.split('/')[-1].split('_')[9].split(':')[1]) - 1

n_examples = 0          # number of examples. Total size of data input. Will be updated.
n_features = 300        # embeding dimension
n_out = 20              # length of output sentence e.g. 20 tokens
n_stop = 2              # how many examples to consider. To use all make it equal n_examples


'''BUILD MODEL FOR TESTING '''
print '\nBuilding model for testing...'
model = load_model(path_model)
infenc, infdec = InfModel(latent_dim=n_units, model=model, encoder_cells=n_enc_cell, decoder_cells=n_dec_cell)
print 'Finished!\n'


''' LOAD EMBEDINGS AND INDEXES'''
print '\nUnPickle emb_dict...'
emb_dict = UnPickleDict(path_emb_dict)
print 'Finished!\n'

print '\nUnPickle data..'
_, titles = UnPickleData(path_data)
y_target = titles[:n_stop]
print 'Finsihed!\n'


''' READING INPUT '''
print '\nReading H5 file to numpys...'
x_emb, _, _, _, _ = H52Np(path_h5)
print 'Finished!\n'



''' PREDICT SEQUENCE'''
print '\nPredicting sequences...'
print x_emb[:n_stop].shape
y_pred = [PredictSequence(infenc, infdec, x_emb.reshape(1,52,300), n_out, n_features) for x_emb in x_emb[:n_stop]]
y_pred = np.array(y_pred)
print y_pred.shape
print 'Finsihed!\n'

# convert preditions to indexes
print '\nConvert preditions to indexes...'
pred_titles = []

for predict, target in zip(y_pred, y_target):
    # get words sequence and append
    print '\tworking on new sequence...'
    print 'Gold: ',target
    pred_title = Emb2Word(predict, emb_dict)
    print 'Predicted: ',pred_title
    pred_titles.append(pred_title)
print 'Finished!\n'




print '\nProgram Finished!'
