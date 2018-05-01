import os
import json
import gzip
from keras.preprocessing.text import text_to_word_sequence

from functions_textsum import PickleData, UnPickleData



''' PATH DATA '''
path_data = './data/signalmedia-1m.jsonl.gz'



path_news = './data/'
path_blog = './data/'

'''
	READS JSON AND TOKENIZE ON WORDS AND DUMP TO PICKLE
'''

n_news = 0 	# News Count
n_blogs = 0 	# Blogs count

# open zip and start reading
with gzip.open(path_data, "rb") as f:
  print '\nOpen JSON to read...'
  
  for line in f:
      
    # convert from JSON
    entry = json.loads(line.decode("ascii"))
    
    ID = entry['id']
    content = entry['content']
    title = entry['title']
    media = entry['media-type']
    source = entry['source']
    published = entry['published']
    
    # clear entry variable
    del entry  
    
    # Do some filtering
    content = content.replace("\n", "").replace("\'s", "")
    # tokenize the content
    content = text_to_word_sequence((content), filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split=' ')
    
    title = title.replace("\n", "").replace("\'s", "")
    # tokenize the title
    title = text_to_word_sequence((title), filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split=' ')
    
    # dump to pickle and append
    PickleData(ID, content, title, media, source, published)
    
    if media == 'News':
      n_news += 1
    else:
      n_blogs += 1
    # clean memory  
    del ID, content, title, media, source, published
      
path_news += 'News_size:%s_ALL.pickle'%(n_news)
path_blog += 'Blog_size:%s_ALL.pickle'%(n_blogs)

os.rename('News.pickle', path_news)
os.rename('Blog.pickle', path_blog)

print "\tFinsihed!\n"
print 'News: ',n_news,'\tBlog: ',n_blogs
 
print '\nProgram Finished!' 
