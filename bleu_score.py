from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import cPickle as pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def UnPickleOut(dataFile):
  target = []           #array of alll target values
  predicted = []        # array of all predicted values
  with (open(dataFile, "rb")) as openfile:
    while True:
      try:
        y, y_hat = pickle.load(openfile)
        target.append(y)
        predicted.append(y_hat)
      except EOFError:
        break
    return target, predicted

filename = './predictions/News_target_predict_50tokens_5size.pickle'

target, predicted = UnPickleOut(filename)

blu = []
for y, y_hat in zip(target, predicted):
	t = [w.encode("utf-8") for w in y]
	p = [w.encode("utf-8") for w in y_hat.T[0]]
	print t
	print p
	blu.append(sentence_bleu(t, p, weights=(1, 0, 0, 0)))
	print('BLEU-1 Cumulative 1-gram: %f' % sentence_bleu(t, p, weights=(1, 0, 0, 0)))

plt.plot(blu) # plotting by columns
media = filename.split('/')[-1].split('_')[0]
size = filename.split('/')[-1].split('_')[3]
filename = './graphs/%s_BLEU_%ssize.png'%(media, size)
plt.savefig(filename)


print '\nProgram Finished!'
