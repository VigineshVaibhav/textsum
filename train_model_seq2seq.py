import sys
from functions_textsum import *

from numpy import array_equal
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.models import load_model
from keras import callbacks
from keras import optimizers


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


path_numpy = './h5files/News_size:25818_tokens:50_type:TRAIN_NUMPYS.h5'


# create different optimizers
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
rms_prop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)       # usualy good RNN
optimizers = {rms_prop:'RMSProp', sgd:'SGD'}

# loss functions
loss_funcs = ['cosine_proximity', 'mean_squared_error', 'mean_absolute_error']
loss = {'cosine_proximity':'COSprox', 'mean_squared_error':'MeanSqarErr', 'mean_absolute_error':'MeanAbsErr'}


''' READING INPUT '''
print '\nReading H5 file to numpys...'
x_emb, x_index, y_emb, y_emb_shifted, y_index = H52Np(path_numpy)
del x_index, y_index		# delete useless data
print 'Finished!\n'


'''DEFINE PARAMETERS '''
flip = 'YES'		    # 'YES' will flip and 'NO' will not flip
data_split = 0.2		# size of data for validation
n_lstm_enc = 0			# number of LSTM cells in encoder set 0 for defautl
n_lstm_dec = 0			# number of LSTM cells in decoder set to 0 for default
n_examples = x_emb.shape[0]	# number of examples. Total size of data input. Will be updated.
n_features = 300		# embeding dimension
n_in = 50			# length of input article e.g. 50 tokens
n_out = 10			# length of output sentence e.g. 20 tokens
n_units = 256			# number of recurrent units inside each LSTM cell
n_epochs = 20			# number of epochs to run
n_stop = None			# how many examples to consider. To use all examples type None
media = path_numpy.split('/')[-1].split('_')[0]	# News or Blog
loss_func = loss_funcs[1]	# loss functions
opt = rms_prop			# optimizer
# final model saved
path_model = './models/%s_size:%s_tokens:%s_unit:%s_flip:%s_epoch:%s_opt:%s_loss:%s_nenc:%s_ndec:%s_FINALMODEL.h5'%(media, n_stop, n_in, n_units, flip, n_epochs, optimizers[opt], loss[loss_func], (n_lstm_enc+1), (n_lstm_dec+1))
# best model saved with callbacks
path_best_model = './models/%s_size:%s_tokens:%s_unit:%s_flip:%s_epoch:%s_opt:%s_loss:%s_nenc:%s_ndec:%s_BESTMODEL.h5'%(media, n_stop, n_in, n_units, flip, n_epochs, optimizers[opt], loss[loss_func], n_lstm_enc, (n_lstm_dec+1))
# graph save path
file_graph = './graphs/%s_size:%s_tokens:%s_unit:%s_flip:%s_epoch:%s_opt:%s_loss:%s_nenc:%s_ndec:%s_GRAPH.png'%(media, n_stop, n_in,  n_units, flip, n_epochs, optimizers[opt], loss[loss_func], n_lstm_enc, n_lstm_dec)
''' DEFINE MODEL '''
train = TrainModel(n_features, n_features, n_units, n_lstm_enc, n_lstm_dec)

train.compile(optimizer=opt, loss=loss_func, metrics=['accuracy'])

# make data more eficient
X1 = x_emb[:n_stop]
X2 = y_emb_shifted[:n_stop]
y = y_emb[:n_stop]
if flip == 'FLIP':
    # flipping output data
    y = FlipNumpy(y) # flip data

# delete unused data
del x_emb, y_emb, y_emb_shifted

''' CREATE CHECKPOINTS '''

best_model = callbacks.ModelCheckpoint(path_best_model, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=1)
# filepath : path and name of file to save.
# verbose : 0 won't print 1 will print on each epoch if improved or not.
# save_best_only : the latest best model according to the quantity monitored will not be overwritten.
# save_weights_only : True - only model weights saved. False - whole model saved model.save(filepath).
# mode : maximization or the minimization of the monitored quantity e.g. val_loss mode=min
# period : how often to cheack. e.g. period=2 ever 2 epoch will check

early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='min')
# monitor 	: can be val_loss or val_acc. We want to decrease val_loss and increase val_acc
# min_delta 	: the minimum value that count as improvement e.g. delta = current_val - previous_val
# patience 	: how many times we allow the model to NOT improve e.g. patience=2 at the second time the model did not improve will STOP
# verbose 	: will print a message of the epoch where it early stopped
# mode 		: which direction we want to go max, min or auto e.g. val_loss we use mode=min val_acc we use mode=max

# When NaN loss is encounter, it will stop
nan_terminate = callbacks.TerminateOnNaN()

#callbacks_list = [best_model, nan_terminate, early_stop]
callbacks_list = [nan_terminate, best_model]


''' TRAIN MODEL '''
print '\nStarted training for %s epochs...'%n_epochs
history = train.fit([X1, X2], y, validation_split=data_split, epochs=n_epochs, callbacks=callbacks_list, verbose=1)

# Save model
print '\nSaving model...'
train.save(path_model)
print 'Finsihed!\n'

print '\nSaving graph...'
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# save to figure
plt.savefig(file_graph)
print 'Finsihed!\n'


print '\nProgram Finished!'
