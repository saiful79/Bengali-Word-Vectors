'''
Training Bengali News Word Vectors
In this script, we will use the data we scraped from news websites to train a Word2Vec model for Bengali.
Then we will test the model to see how well it is performing.
First we import the packages we need

i follow the link 
# https://www.kaggle.com/csoham/classification-bengali-news-articles-indicnlp/downloads/News%20Articles.zip/1
'''


import json
import os
import re
import string
import numpy as np
from gensim.models import Word2Vec
import glob
import time
from gensim.models import KeyedVectors
from gensim.test.utils import datapath

'''

Let's define a function that will read the data file and extract the fields we want.

In our case, we will be using the article body for training

'''

def get_body_text(filename, field):
	count = 0
	extracted_body_field=[]
	
	with open(os.path.join(filename), 'r') as f:
	
		articles=json.load(f)

	for article in articles['articles']:
		count+=1
	
		# print(article[field])
		# line = re.sub("[^A-Za-z]", "", article[field].strip())
		# myString = line

		extracted_body_field.append(article[field].strip())
	print("total data",count)
	return extracted_body_field


'''
Now we define a function to preprocess our data.
The function does the following:
It replaces common texts found in the data and replaces that with our custom text
It removes all emoji's and emoticons from the text
It removes all English text

'''

def replace_strings(texts, replace):
    new_texts=[]
    
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    english_pattern=re.compile('[a-zA-Z0-9]+', flags=re.I)
    
    for text in texts:
        for r in replace:
            text=text.replace(r[0], r[1])
        text=emoji_pattern.sub(r'', text)
        text=english_pattern.sub(r'', text)
        text=re.sub(r'\s+', ' ', text).strip()
        new_texts.append(text)

    return new_texts

'''
We also need to remove all the punctuations in our data. The remove_pun function removes all common punctuations found in text.
'''
def remove_punc(sentences):
    # import ipdb; ipdb.set_trace()
    new_sentences=[]
    exclude = list(set(string.punctuation))
    exclude.extend(["’", "‘", "—"])
    for sentence in sentences:
        s = ''.join(ch for ch in sentence if ch not in exclude)
        new_sentences.append(s)
    
    return new_sentences

'''
Now that we have our preprocessed training data, we can start training our model.

We will generate embeddings for each word of size 200 and use 5 words in its vicinity to figure out the meaning of the word
'''
def word_tokenization(total_text_list):

	body=[article.split('।') for article in total_text_list]
	body=[item for sublist in body for item in sublist]
	body=[item.strip() for item in body if len(item.split())>1]

	body=[item.split() for item in body]

	model = Word2Vec(body[:10], size=200, window=5, min_count=1)
	model_name = "word2vec"
	# model.save(model_name+".model")
	model.wv.save_word2vec_format(model_name+'.bin', binary=True)
	model = KeyedVectors.load_word2vec_format(model_name+'.bin', binary=True)
	model.save_word2vec_format(model_name+'.txt', binary=True)

	print(body[:10])

if __name__=="__main__":
	replace=[('\u200c', ' '),
			('\u200d', ' '),
			('\xa0', ' '),
			('\n', ' '),
			('\r', ' ')]

	path = glob.glob("NewsArticles/*.json")
	total_text = []
	print(path)
	for txt_file in path:
		extract_text = get_body_text(txt_file,'body')
		process_text =remove_punc(extract_text)
		process_text=replace_strings(process_text, replace)

		total_text.append(process_text)

	
	all_sentence = []
	for i in total_text:
		for j in i:
			all_sentence.append(j)

	print("total train data : ",len(all_sentence))
	word_tokenization(all_sentence)
