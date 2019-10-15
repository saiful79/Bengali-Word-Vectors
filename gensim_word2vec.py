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

# def get_marge_all_text(text_dic):
# 	total_text=[]
# 	i=0
# 	for article in text_dic['text_field']:
# 		i+=1
# 		total_text.extend(article)
# 	print("marge text list : ",i)


def word_tokenization(total_text_list):
	body=[article.split('।') for article in total_text_list]
	body=[item for sublist in body for item in sublist]
	body=[item.strip() for item in body if len(item.split())>1]

	body=[item.split() for item in body]

	print(body[:10])

if __name__=="__main__":
	replace=[('\u200c', ' '),
			('\u200d', ' '),
			('\xa0', ' '),
			('\n', ' '),
			('\r', ' ')]

	text_dic={}

	path = glob.glob("NewsArticles/*.json")

	print(path)
	for txt_file in path:
		extract_text = get_body_text(txt_file,'body')
		process_text =remove_punc(extract_text)
		process_text=replace_strings(process_text, replace)

		text_dic['text_field']= process_text

	# get_marge_all_text(text_dic)
	with open('file.txt', 'w') as file:
		file.write(str(text_dic))

	# print("data type",len(extract_text))

	# # print("\x1b[31mCrawled Unprocessed Text\x1b[0m")
	# print(extract_text[30])



	# ebala_body=remove_punc(extract_text)

	# print("Sentences after removing all punctuations")
	# print(ebala_body[1])

	# ebala_body=replace_strings(ebala_body, replace)

	# print("Sentences after replacing strings")
	# print(ebala_body[1])

	# abz_body=get_body_text('NewsArticles/anandabazar_articles.json', 'body')

	# abz_body=remove_punc(abz_body)
	# abz_body=replace_strings(abz_body, replace)

	# zee_body=get_body_text('NewsArticles/zeenews_articles.json', 'body')

	# zee_body=remove_punc(zee_body)
	# zee_body=replace_strings(zee_body, replace)

	# body=[]
	# body.extend(zee_body)
	# body.extend(abz_body)
	# body.extend(ebala_body)

	# print(f"Total Number of training data: {len(body)}")

	# word_tokenization(body)