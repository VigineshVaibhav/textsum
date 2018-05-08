#imports
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
import cPickle as pickle
import h5py

'''------------------DEFINE FUNCTIONS-----------------------------'''

def TrainModel(n_input, n_output, n_units, encoder_cells=0, decoder_cells=0, activation='tanh'):
  """ CREATE MODEL FOR TRAINING

  Args:
      n_input       (int): number of input sequence, e.g. number of features, words, or characters for each time step.
      n_output      (int): number of output sequence, e.g. number of features, words, or characters for each time step.
      n_units       (int): number of cells to create in the encoder and decoder models, e.g. 128 or 256.
      encoder_cells (int): number of LSTM cells in encoder
      decoder_cells (int): number of LSTM cells in decoderstate
      activ_func    (str): stirng wither softmax or tanh depending on model. By default is tanh

  Returns:
      model_train (model): model that can be trained given source, target, and shifted target sequences.

  """

  # NOTE: The model is trained given source and target sequences where the model takes both
  # the source and a shifted version of the target sequence as input and predicts the whole target sequence.

  # define training encoder
  encoder_inputs = Input(shape=(None, n_input), name='ENCODER_INPUT')

  # add required number of LSTM cells
  if encoder_cells > 0:
    encoder = LSTM(n_units, return_sequences=True, return_state=False, name='ENCODER_LSTM_0')(encoder_inputs)
    for i in range(encoder_cells-1):
      encoder = LSTM(n_units, return_sequences=True, return_state=False, name='ENCODER_LSTM_%s'%(i+1))(encoder)
    encoder = LSTM(n_units, return_state=True, name='ENCODER_LSTM_%s'%encoder_cells)(encoder)
  else:
    encoder = LSTM(n_units, return_state=True, name='ENCODER_LSTM_0')(encoder_inputs)

  encoder_outputs, state_h, state_c = encoder
  encoder_states = [state_h, state_c]

  # NOTE: we return 3 because we use both states of LSTM as input to decoder!
  # state_h : hidden state
  # state_c : cell state

  # define training decoder
  decoder_inputs = Input(shape=(None, n_output), name='DECODER_INPUT')

  # add required number of LSTM cells
  if decoder_cells > 0:
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True, name='DECODER_LSTM_0')
    decoder_lstm = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    for i in range(decoder_cells-1):
      decoder_lstm = LSTM(n_units, return_sequences=True, return_state=False, name='DECODER_LSTM_%s'%(i+1))(decoder_lstm)
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True, name='DECODER_LSTM_%s'%decoder_cells)(decoder_lstm)

  else:
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True, name='DECODER_LSTM_0')
    decoder_lstm = decoder_lstm(decoder_inputs, initial_state=encoder_states)

  decoder_outputs, _, _ = decoder_lstm
  decoder_dense = Dense(n_output, activation=activation, name='DECODER_OUTPUT')
  decoder_outputs = decoder_dense(decoder_outputs)

  # create model
  model_train = Model([encoder_inputs, decoder_inputs], decoder_outputs)

  return model_train


def InfModel(latent_dim, model, encoder_cells, decoder_cells):
  """ CREATE  INFERENCE MODEL FOR TESTING

  Args:
      latent_dim      (int): length of weights inside of LSTM
      model         (model): trained model that can be loaded from .h5 file
      encoder_cells   (int): number of LSTM cells in encoder
      decoder_cells   (int): number of LSTM cells in decoder

  Returns:
      inf_enc       (model): inference encoder model.
      inf_enc       (model): inference decoder model.

  """

  # NOTE: We need to build an inference model for decoder and encoder. States of encoder
  # will be imported to state of decoder


  encoder_inputs = model.layers[0].input
  _, state_h, state_c = model.layers[encoder_cells + 2].output

  encoder_states = [state_h, state_c]

  # inference encoder
  inf_enc = Model(encoder_inputs, encoder_states)

  decoder_inputs = model.layers[encoder_cells + 1].input
  decoder_lstm = model.layers[encoder_cells + 3]

  decoder_dense = model.layers[-1]

  decoder_state_input_h = Input(shape=(latent_dim,))
  decoder_state_input_c = Input(shape=(latent_dim,))

  decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

  decoder_lstm = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)

  if decoder_cells > 0:
    for i in range(decoder_cells):
      decoder_lstm = model.layers[encoder_cells + 4 + i](decoder_lstm)

  decoder_outputs, state_h, state_c = decoder_lstm
  decoder_states = [state_h, state_c]

  decoder_outputs = decoder_dense(decoder_outputs)

  # inference decoder
  inf_dec = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

  return inf_enc, inf_dec


def PredictSequence(inf_encoder, inf_decoder, source, n_steps, n_output):
  """ CREATE MODELS FOR TRAINING AND INFERENCE

  Args:
      inf_encoder (model): inference encoder model used when making a prediction for a new source sequence.
      inf_decoder (model): inference decoder model used when making a prediction for a new source sequence.
      source     (matrix): encoded source sequence.
      n_steps       (int): number of time steps in the target sequence.
      n_output      (int): length ofoutput sequence, e.g. the number of features, words, or characters for each time step.

  Returns:
      output      (array): list containing the target sequence.

  """

  # encode
  state = inf_encoder.predict(source)

  # start of sequence input
  target_seq = array([0.0 for _ in range(n_output)]).reshape(1, 1, n_output)

  # collect predictions
  output = list()

  for t in range(n_steps):

    # predict next sequence
    yhat, h, c = inf_decoder.predict([target_seq] + state)

    # store prediction
    output.append(yhat[0,0,:])

    # update state
    state = [h, c,]

    # update target sequence
    target_seq = yhat

  return array(output)


# dump to pickle
def PickleData(ID, content, title, media, source, published, identifier=''):
  ''' IF WE NEED AN IDENTIFIER TO DISTINGUISH BETWEEN TRIAN AND TEST '''
  if identifier == '':
  	with open('%s.pickle'%(media), 'a+b') as handle:
    		pickle.dump([ID, content, title, media, source, published], handle, -1)
  else:
	 with open('%s_%s.pickle'%(media, identifier), 'a+b') as handle:
		pickle.dump([ID, content, title, media, source, published], handle, -1)
  return


# retrieve form pickle
def UnPickleData(filename):
  with (open(filename, "rb")) as openfile:
    i = 0
    contents = []
    titles = []
    while True:
      try:
        _, content, title, _ , _ , _ = pickle.load(openfile)
        contents.append(content)
        titles.append(title)
        del content, title
      except EOFError:
        break
    return contents, titles


# Pickle any dictionary
def PickleDict(filepath, value):
        with open(filepath, 'wb') as handle:
                pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return


# Unpickle any  dictionary
def UnPickleDict(filepath):
        with open(filepath, 'rb') as handle:
                output = pickle.load(handle)
        return output

# Function takes input data and outputs numpy_emb numpy_index
def NumpyData(data, word_index, emb_dict, n_input, n_emb, s_flag=False, e_flag=False):
    '''
        Get embedding and index only if exists
        If any of them don't exist replace with <UNK> embedding and index
    '''
    data_size = len(data)
    n_words = len(word_index.items)
    # Create empty numpy arrays
    np_emb = np.zeros((data_size, (n_input + 2), n_emb), dtype=float)
    np_hot = np.zeros((data_size, (n_input + 2), n_words), dtype=int)
    offset = 0      # offset use when adding flag

    # ia - index of article
    # iw - index for word of each article
    for ia, article in enumerate(data):
        if s_flag:
            # Add start flag
            np_emb[ia][0] = emb_dict['<EOS>']
            np_hot[ia][0] = to_categorical(word_index['<EOS>'], num_classes=n_words)
            offset = 1

        for iw, word in enumerate(article):
            if word in emb_dict:
                # word in embedding
                np_emb[ia][iw + offset] = emb_dict[word]
            else:
                # we deal with unknown words
                np_emb[ia][iw + offset] =emb_dict['<UNK>']
            if word in word_index:
                # check word_index
                np_hot[ia][iw + offset] = to_categorical(word_index['<UNK>'], num_classes=n_words)
            else:
                # we deal with unknown words
                np_hot[ia][iw + offset] = to_categorical(word_index['<UNk>'], num_classes=n_words)

        if e_flag:
            # Add end flag
            np_emb[ia][len(article)+1] = emb_dict['<EOS>']
            np_hot[ia][0] = to_categorical(word_index['<EOS>'], num_classes=n_words)

        return np_emb, np_index


# Shift matrix by 1
def ShiftRight(np_input):
        # np_output - output of the numpy. same shape as input
        # i_s : index sequence
        # seq : sequence of input
        # i_e : size of each embedding in the sequence
        # create shifted output values
        np_output = np.zeros(np_input.shape, dtype=float)
        for i_s,seq in enumerate(np_input):
                seq_size = len(seq)
                for i_e in range(seq_size - 1):
                        # shift eahc line by 1 and leave  0
                        np_output[i_s][i_e + 1] = seq[i_e]
        return np_output

# Flip any numpy and keep 0 padding
def FlipNumpy(np_in):
    # flip numpy on axis 1
    np_flip = np.flip(np_in, 1)
    np_out = np.zeros(np_in.shape)
    np_zeros = np.zeros(np_in.shape[-1])
    for i_l, line in enumerate(np_flip):
        offset = 0
        for i_v, value in enumerate(line):
            if np.array_equal(value, np_zeros):
                offset += 1
                continue
            else:
                np_out[i_l][i_v - offset] = value
    del np_flip
    return np_out


def Np2H5(filename, x_emb, x_index, y_emb, y_index):
	'''
	SAVE NUMPYS TO HDF5 FORMAT

	Args:
		filename (string): name and path of .h5 file
		x_emb (numpy): input numpy
		x_index
		y_emb
		y_emb_shifted
		y_index
	Returns:

	'''
	print '\nFinished saving:'
	hf = h5py.File(filename, 'w')
	hf.create_dataset('x_emb', data=x_emb)
	print 'x_emb'
	hf.create_dataset('x_index', data=x_index)
	print 'x_index'
	hf.create_dataset('y_emb', data=y_emb)
	print 'y_emb'
	hf.create_dataset('y_index', data=y_index)
	print 'y_index'
	hf.close()
	print 'hdf5 closed!\n'
	return


def H52Np(filename):
	'''
        SAVE NUMPYS TO HDF5 FORMAT

        Args:
                filename 	(string): name and path of .h5 file

	Returns:
                x_emb 		(numpy): input numpy
                x_index 	(numpy): input numpy
                x_emb 		(numpy): input numpy
                y_index 	(numpy): target output numpy

	'''
	hf = h5py.File(filename, 'r')	# read .h5 file
	keys = hf.keys()		# keys of each saved numpy
	if len(keys) != 4:
		print 'Wrong .h5 files!'
		return
	x_emb 		= hf.get(keys[0])
	x_index 	= hf.get(keys[1])
	y_emb 		= hf.get(keys[2])
	y_index 	= hf.get(keys[3])
	return x_emb, x_index, y_emb, y_index


def Cosim(a, b):
        '''
        APPROXIMATE WORD BASED ON COSIMILARITY OF EMBEDINGS

        Args:
                a,b (vector): input vectors, Can be any size
        Returns:
                cosimilarity (int): cosimilarity between 2 vectors
        '''

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)


def Emb2Word(array_emb, emb_dict):
        '''
        APPROXIMATE WORD BASED ON COSIMILARITY OF EMBEDINGS

        Args:
                array_emb (vector): vector embeding of a word
                emb_dict (dictionary): dictionary of word embedings. Keys: words Values: embedings
                word_index (dicitonary): dictionary of indexes: Keys: word Values: index
        Returns:
                word (string): possibl word of inputed embeding

        '''
        sim_word = {}   # most similar word
        n_input = len(array_emb)
        sim =[[0,''] for _ in range(n_input)]   # cosimilarity is between 0-1. Most similar is 1

        for word, emb in emb_dict.items():
                cosim = [Cosim(array_emb, emb)]
                sim = [[y,word] if x[0] < y else x for x,y in zip(sim,cosim)]

        return [w[1] for w in sim]	# returns word




