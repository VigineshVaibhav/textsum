import sys
from functions_textsum import UnPickleData
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

#filename = './predictions/News_target_predict_50tokens_5size.pickle'

#target, predicted = UnPickleOut(filename)


path_data = './data/News_size:10545_tokens:50_type:TEST_DATA.pickle'

media = path_data.split('/')[-1].split('_')[0]
size = path_data.split('/')[-1].split('_')[1].split(':')[1]

contents, titles = UnPickleData(path_data)

predicts = []
print '\nPredicting with baseline...'
for content, title in zip(contents, titles):
    predicts.append(content[:10])
print 'Finsihed!\n'
#sys.exit()

print '\nComputing BLEU score for baseline...'
blu = []
for y, y_hat in zip(titles, predicts):
	t = [w.encode("utf-8") for w in y]
	p = [w.encode("utf-8") for w in y_hat]
#	print t
#	print p
	blu.append(sentence_bleu(t, p, weights=(1, 0, 0, 0)))
#	print('BLEU-1 Cumulative 1-gram: %f' % sentence_bleu(t, p, weights=(1, 0, 0, 0)))
print 'Finsihed!\n'

print '\nAVERAGE BLEU: %s'%(sum(blu)/len(blu))

plt.plot(blu) # plotting by columns
filename = './graphs/%s_BLEU_%ssize_CUM1Ngram.png'%(media, size)
plt.savefig(filename)


print '\nProgram Finished!'
