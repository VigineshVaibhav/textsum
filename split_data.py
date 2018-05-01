import os
import numpy as np
import cPickle as pickle

from functions_textsum import PickleData


''' SET PARAMETERS  '''

news_split = 100000
blog_split = 40000

path_news = 	'./data/News_size:734488_ALL.pickle'          # only fill these
path_blog = 	'./data/Blog_size:265512_ALL.pickle'         # only fill these

path_split = './data/'				# where do you want save splits

''' ADITIONSL FUNCTIONS FOR SPLITTING'''
# create the pickle with 400 - 20 tokens
def UnPickle400(path_file):
	count = 0	# counter for the 400
	media_type = path_file.split('/')[-1].split('_')[0]	# get media type from path +name
	with (open(path_file, "rb")) as openfile:
		while True:
			try:
				ID, content, title, media, source, published = pickle.load(openfile)
				n_c = len(content)
				n_t = len(title)
				if (n_c <= 400) and (n_t <= 10):
					# dump it to pickle
					PickleData(ID, content, title, media, source, published)
					count += 1
				# cleab memory
				del ID, content, title, media, source, published, n_c, n_t
			except EOFError:
				break
	#rename file
	path = path_split + '%s_size:%s_tokens:400_DATA.pickle'%(media_type, count)
	os.rename('%s.pickle'%(media_type), path)
	# return path of created file
	return path

# Unpickle to what ever length we want smaller than 400
def UnPickleTo(path_file, n_tokens):
	partition = path_file.split('/')[-1].split('_')[3].split(':')[1]	# train or test
	count = 0       # counte
        media_type = path_file.split('/')[-1].split('_')[0]      # get media type from path +name
        with (open(path_file, "rb")) as openfile:
                while True:     
                        try:    
                                ID, content, title, media, source, published = pickle.load(openfile)
                                n_c = len(content)
                                if (n_c <= n_tokens):
                                        # dump it to pickle
                                        PickleData(ID, content, title, media, source, published, identifier=n_tokens)
                                        count += 1
                                # cleab memory
                                del ID, content, title, media, source, published, n_c
                        except EOFError:
                                break
        #rename file
	path = path_split + '%s_size:%s_tokens:%s_type:%s_DATA.pickle'%(media_type, count, n_tokens, partition)
        os.rename('%s_%s.pickle'%(media_type, n_tokens), path)
        return


# Unpickle data to trian and test
def UnPickleSplit(path_file, test_size):
	total = int( path_file.split('/')[-1].split('_')[1].split(':')[1])	# get total numbe rof files
	train_size = total - test_size
	index = 0					# keep track of current index
	split = np.random.randint(total, size=test_size)   # splitting numbers
	media_type = path_file.split('/')[-1].split('_')[0]	# get media or blog
	n_tokens = path_file.split('/')[-1].split('_')[2].split(':')[1]

	with (open(path_file, "rb")) as openfile:
		while True:
			try:
				ID, content, title, media, source, published = pickle.load(openfile)

				if index in split:
					# add to test data
					PickleData(ID, content, title, media, source, published, identifier='TEST')
				else:
					# add to train data
					PickleData(ID, content, title, media, source, published, identifier='TRAIN')

				# increment index
				index += 1

			except EOFError:
				break
	path_train = path_split + '%s_size:%s_tokens:%s_type:TRAIN_DATA.pickle'%(media_type, train_size, n_tokens) 
	path_test = path_split + '%s_size:%s_tokens:%s_type:TEST_DATA.pickle'%(media_type, test_size, n_tokens)

	os.rename('%s_TRAIN.pickle'%(media_type), path_train)
	os.rename('%s_TEST.pickle'%(media_type), path_test)

	# return paths
	return path_train, path_test




# paths automatic generated
path_news400 = ''
path_blog400 = ''

path_news400_test = ''
path_blog400_test = ''

path_news400_train = ''
path_blog400_train = ''


def Create400Pickle():
	# get 400 tokens partition data
	print '\nCreating Pickle of all 400 token length articles...'
	
	global path_news400
	path_news400 = UnPickle400(path_news)
	
	print '\tDone with News', path_news400
	
	global path_blog400
	path_blog400 = UnPickle400(path_blog)
	
	print '\tDone with Blog', path_blog400
	print 'Finished!\n'
	
	return 


def Split400Pickle():
	# Split train + test
	print '\nSpliting for Train and Test...'
	
	global path_news400_train, path_news400_test
	path_news400_train, path_news400_test = UnPickleSplit(path_news400, news_split)
	
	print '\tDone with News!'
	
	global path_blog400_train, path_blog400_test
	path_blog400_train, path_blog400_test = UnPickleSplit(path_blog400, blog_split)
	
	print '\tDone with Blog'
	print 'Finished!\n'
	
	return


def SplitNews():
	# Split to different n_tokens data
	print '\nUnpickle News to \n\t50 tokens'

	UnPickleTo(path_news400_train, 50)
	UnPickleTo(path_news400_test, 50)

	print '\t200 tokens'

	UnPickleTo(path_news400_train, 200)
	UnPickleTo(path_news400_test, 200)

	return


def SplitBlog():
	print 'Unpickle Blog to\n\t50 tokens'

	UnPickleTo(path_blog400_train, 50)
	UnPickleTo(path_blog400_test, 50)

	print '\t200 tokens'

	UnPickleTo(path_blog400_train, 200)
	UnPickleTo(path_blog400_test, 200)

	return


Create400Pickle()
Split400Pickle()
SplitNews()
SplitBlog()


print '\nProgram Finished!'
