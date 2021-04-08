## Advanced NLP and Accounting/Finance

import os
import gensim
import spacy
import nltk
from nltk.corpus import stopwords

nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
nlp.max_length = 5000000

stop_words = stopwords.words("english")
stop_words.extend(['proc','type','mic','clear','originator','name','webmaster','www','gov','originator',
                   'key','asymmetric','dsgawrwjaw','snkk','avtbzyzmr','agjlwyk','xmzv','dtinen','twsm',
                   'sdw','oam','tdezxmm','twidaqab','mic','info','rsa','md','rsa','kn','ln','cgpgyvylqm',
                   'covp','srursowk','xqcmdgb','mdyso','zjlcpvda','hx','lia','form','period','ended',])

source_dir = 'D:/Data/Reasonable_10K/'

def iter_documents(source_dir):
    i=1
    for root, dirs, files in os.walk(source_dir):
        for fname in files:
            document = open(os.path.join(root, fname)).read().split('</Header>')[1]
            tokens = gensim.utils.simple_preprocess(document)
            red_tokens = [token for token in tokens if token not in stop_words]
            doc = nlp(" ".join(red_tokens))
            lemmas = [token.lemma_ for token in doc if token.pos_ in ['NOUN']]
            print(str(i),end = '\r',flush=True)
            i+=1
            yield lemmas

files_to_lemmas = iter_documents(source_dir)

dictionary = gensim.corpora.Dictionary(files_to_lemmas)

import pickle

# Save dump file

fp = open('dictionary.rdata','wb')
pickle.dump(dictionary,fp)
fp.close()

# Load dump file

fp = open('dictionary.rdata','rb')
dictionary = pickle.load(fp)
fp.close()

dictionary.filter_extremes(no_below=10, no_above=0.5)

files_to_lemmas = iter_documents(source_dir)

corpus = [dictionary.doc2bow(lemmas) for lemmas in files_to_lemmas]

# Save dump file
fp = open('corpus.rdata','wb')
pickle.dump(dictionary,fp)
fp.close()

# Load dump file
fp = open('corpus.rdata','rb')
dictionary = pickle.load(fp)
fp.close()

lda_model = gensim.models.ldamulticore.LdaModel(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=24, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=len(corpus)/20,
                                           passes=100,
                                           alpha='asymmetric',
                                           per_word_topics=False,
                                            minimum_probability=.0,
                                            eta = 'auto')

# Save model
lda_model.save('first_model.lda')

# Load model
gensim.models.ldamulticore.LdaModel.load('first_model.lda')

import pyLDAvis.gensim

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)

vis

import pandas as pd

top_words_df = pd.DataFrame()
for i in range(24):
    temp_words = lda_model.show_topic(i,10)
    just_words = [name for (name,_) in temp_words]
    top_words_df['Topic ' + str(i+1)] = just_words

top_words_df.T.to_csv('topics.csv')

import datetime

from dateutil.parser import parse

files = os.listdir(source_dir)

file_dates = [parse(item.split('_')[0]) for item in files]

import numpy as np

evolution = np.zeros([len(corpus),25])
ind = 0
for bow in corpus:
    topics = lda_model.get_document_topics(bow)
    for topic in topics:
        evolution[ind,topic[0]] = topic[1]
    ind+=1

evolution_df = pd.DataFrame(evolution)
evolution_df['Date'] = file_dates
evolution_df.set_index('Date',inplace=True)

evolution_df

import matplotlib.pyplot as plt

plt.style.use('bmh')

fig,axs = plt.subplots(6,4,figsize = [20,15])
for ax,column in zip(axs.flat,evolution_df.groupby('Date').mean().columns):
    ax.plot(evolution_df.resample('Y').mean()[column])
    ax.set_title('Topic ' + str(column+1),{'fontsize':14})
plt.subplots_adjust(hspace=0.4)
plt.savefig('topic_trends.png',facecolor='white')

evolution_df.groupby('Date').mean()

evolution_df

import collections

files[0]

sample_doc = open(source_dir+files[7000]).read().split('</Header>')[1]

tokens = gensim.utils.simple_preprocess(sample_doc)

red_tokens = [token for token in tokens if token not in stop_words]

collections.Counter(red_tokens).most_common()

doc = nlp(" ".join(red_tokens))
lemmas = [token.lemma_ for token in doc if token.pos_ in ['NOUN']]