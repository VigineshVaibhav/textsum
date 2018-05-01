#imports

from functions_seq2seq import *		# import implemented functions 

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

import pydot
from IPython.display import Image
import time




# configure problem
n_features = 50 + 1
n_steps_in = 6
n_steps_out = 3

# generate a sequence of random integers
def generate_sequence(length, n_unique):
  return [randint(1, n_unique-1) for _ in range(length)]

# prepare data for the LSTM
def get_dataset(n_in, n_out, cardinality, n_samples):
        X1, X2, y = list(), list(), list()
        for _ in range(n_samples):
                # generate source sequence
                source = generate_sequence(n_in, cardinality)
                # define padded target sequence
                target = source[:n_out]
                target.reverse()
                # create padded input target sequence
                target_in = [0] + target[:-1]
                # encode
                src_encoded = to_categorical([source], num_classes=cardinality)
                tar_encoded = to_categorical([target], num_classes=cardinality)
                tar2_encoded = to_categorical([target_in], num_classes=cardinality)
                # store
                X1.append(src_encoded)
                X2.append(tar2_encoded)
                y.append(tar_encoded)
        X1 = np.squeeze(array(X1), axis=1) 
        X2 = np.squeeze(array(X2), axis=1) 
        y = np.squeeze(array(y), axis=1) 
        return X1, X2, y

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]


# generate training dataset
X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 100000)
print(X1.shape,X2.shape,y.shape)

print('X1=%s, X2=%s, y=%s\n\n' % (one_hot_decode(X1[0]), one_hot_decode(X2[0]), one_hot_decode(y[0])))

# define model
train = TrainModel(n_features, n_features, 128, 0, 0)
train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# infenc, infdec = define_models(n_features, n_features, 128)


# train model
s = time.time()
train.fit([X1, X2], y, epochs=1, verbose=1)
e = time.time()

print '\n\nTIME: ',(e-s) #95.4 seconds

# Save model
train.save('s2s_2_2.h5')



model = load_model('s2s_2_2.h5')

infenc, infdec = InfModel(latent_dim=128, model=model, encoder_cells=0, decoder_cells=0)

# evaluate LSTM
total, correct = 100, 0
for _ in range(total):
	X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
	target = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)
	if array_equal(one_hot_decode(y[0]), one_hot_decode(target)):
		correct += 1
    
print('Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))
# spot check some examples
for _ in range(10):
	X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
	target = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)
	print('X=%s y=%s, yhat=%s' % (one_hot_decode(X1[0]), one_hot_decode(y[0]), one_hot_decode(target)))


print '\nProgram Finsihed!'
