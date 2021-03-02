### Basic LDA example - An analysis of ~200 scientific articles
A visual explanation of topic modelling --> https://en.wikipedia.org/wiki/Topic_model

#### Libraries
**Os** is for operating system routines.  
**Pdfminer** is needed for stripping the text from pdf documents.  
**Gensim** includes the LDA functions. State-of-the-art NLP library.  
**Nltk** is needed for stopwords. Basic text manipulation functions.  
**Spacy** is needed for lemmatization. Uses GloVe word embeddings. --> https://nlp.stanford.edu/projects/glove/  
**Pickle** is for saving intermediate steps to the disk.  
**pyLDAvis** is for visualising LDA results.  
**Matplotlib** plotting library  
**Pygam** is for generalized additive models (which are a generalisation of linear regression)  
**Numpy** data manipulation  
**Pandas** data manipulation. The most important Python library if you want to be a data scientist  
**Re** Regular expressions --> https://en.wikipedia.org/wiki/Regular_expression  

import os

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
# From PDFInterpreter import both PDFResourceManager and PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice
# Import this to raise exception whenever text extraction from PDF is not allowed
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.layout import LAParams, LTTextBox, LTTextLine
from pdfminer.converter import PDFPageAggregator

import gensim
from nltk.corpus import stopwords
import spacy 
import gensim.corpora as corpora
import pyLDAvis.gensim
import pickle
from pygam import LinearGAM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from gensim.models import CoherenceModel

#### Stripping text from pdf files
The function below is used for stripping the text from the pdf files. Below is an example of pdf file and the raw text stripped from it.
Pdfminer is not the most easiest pdf-stripper! There are easier options online. Also tools that can be used with a browser.

def convert_pdfminer(fname):
        fp = open(fname, 'rb')
        parser = PDFParser(fp)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        text = ''
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)
            layout = device.get_result()
            for lt_obj in layout:
                if isinstance(lt_obj, LTTextBox) or isinstance(lt_obj, LTTextLine):
                    text += lt_obj.get_text()
        return text

---
Define the data path.

# Different path, different computer
#data_path = 'E:/Onedrive_uwasa/OneDrive - University of Vaasa/Data/Yassin_BD_strategy/articles/'
data_path = 'C:/Users/mran/OneDrive - University of Vaasa/Data/Yassin_BD_strategy/articles/'

---
List pdf files in the destination directory **data_path**

files = os.listdir(data_path)
" | ".join(files[0:10])

---
Load the stopwords from the NLTK library. Stopwords are useless words in the text that does not contain any information. At least information that is important in topic modeling. Very often you need to additional words to the stopwords. These are found by experimenting with the finished LDA model and identifying useless words from the topics. Below are listed the stopwords included in NLTK. More accounting/finance oriented stopword lists can be found from https://sraf.nd.edu/textual-analysis/resources/#StopWords

stop_words = stopwords.words("english")
stop_words.extend(['ieee','cid','et','al','pp','vol','fig','reproduction','prohibited','reproduced','permission'])
" | ".join(stop_words)

---
Collect the publicatin years from the file names using regular expressions. https://www.rexegg.com/regex-quickstart.html

file_years = [re.findall(r'\d+',name)[0] for name in files]
" | ".join(file_years)

---
Go through files and strip text from the pdfs. Takes a lot of time!

raw_text = []
for file in files:
    temp1 = convert_pdfminer(data_path+file)
    raw_text.append(temp1)

raw_text[0][0:1000]

---
Some of the pdf files were scanned documents. Pdfminer cannot strip text from these documents. You need to use a software capable of *optical character recoqnition*. These problematic pdfs were analysed elsewhere and the txt-file were added manually.

# ADD PROBLEMATIC DOCUMENTS
problem_path = 'C:/Users/mran/OneDrive - University of Vaasa/Data/Yassin_BD_strategy/problem_docs/'
problem_files = ['Cook et al 1998.txt',
 'Davenport et al., 2012.txt',
 'Dennis et al., 1997.txt',
 'Kiron et al., 2012.txt',
 'Lin & Hsu, 2007.txt',
 'Sawyerr et al 2003.txt',
 'Yasai-Ardekani & Nystrom, 1996 .txt']
for file in problem_files:
    file1 = open(problem_path + file,'r', encoding='ansi')
    raw_text.append(file1.read())

---
Add the years of the problematic documents to the *file_years* list.

# Years of problem documents
file_years.append('1998')
file_years.append('2012')
file_years.append('1997')
file_years.append('2012')
file_years.append('2007')
file_years.append('2003')
file_years.append('1996')

---
Below I tried to collect the publications dates of the documents from the metadata (file creation date). But this did not work, because the file creation date did not coincide with the original publication date.

"""
import re
filedates = []
missing = []
for item in meta:
    try:
        temp = item[0]['CreationDate']
        temp2 = re.findall(r'\d+',str(temp))
        filedates.append(temp2[0])
    except:
        missing.append"""

---
Save the stripped raw text to the disk.

# Dump list of raw text documents to a file
fp = open('raw_text_list','wb')
pickle.dump(raw_text,fp)
fp.close()

---
Load the raw text from the disk.

# Load dump file
fp = open('raw_text_list','rb')
raw_text = pickle.load(fp)
fp.close()

---
Simple preprocessing using a Gensim function. Below is the function description: 

Convert a document into a list of lowercase tokens, ignoring tokens that are too short or too long.  
Parameters  
*doc (str)* – Input document.  
*deacc (bool, optional)* – Remove accent marks from tokens  
*min_len (int, optional)* – Minimum length of token (inclusive). Shorter tokens are discarded.  
*max_len (int, optional)*– Maximum length of token in result (inclusive). Longer tokens are discarded.  

docs_cleaned = []
for item in raw_text:
    tokens = gensim.utils.simple_preprocess(item)
    docs_cleaned.append(tokens)

" | ".join(docs_cleaned[0][0:200])

---
Remove stopwords from the texts

docs_nostops = []
for item in docs_cleaned:
    red_tokens = [word for word in item if word not in stop_words]
    docs_nostops.append(red_tokens)

" | ".join(docs_nostops[0][0:200])

---
Spacy is a multi-purpose NLP tool that can be used, for example, to named-entity-recognition, parts-of-speech tagging, lemmatisation etc. Examples of these methods --> https://stanfordnlp.github.io/CoreNLP/  
We need here only the lemmatisation capabilties. Also we define that only nouns,adjectives,verbs and adverbs are saved. All other words are discarded. For these operations we use the medium size pretrained model.
Lemmatisation explanation from Wikipedia:  
* Lemmatisation (or lemmatization) in linguistics is the process of grouping together the inflected forms of a word so they can be analysed as a single item, identified by the word's lemma, or dictionary form. In computational linguistics, lemmatisation is the algorithmic process of determining the lemma of a word based on its intended meaning. Unlike **stemming**, lemmatisation depends on correctly identifying the intended part of speech and meaning of a word in a sentence, as well as within the larger context surrounding that sentence, such as neighboring sentences or even an entire document.  
The word "better" has "good" as its lemma. This link is missed by stemming, as it requires a dictionary look-up.  
The word "walk" is the base form for word "walking", and hence this is matched in both stemming and lemmatisation.

allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']

nlp = spacy.load('en_core_web_md', disable=['parser', 'ner'])

docs_lemmas = []
for red_tokens in docs_nostops:
    doc = nlp(" ".join(red_tokens))
    docs_lemmas.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

" | ".join(docs_lemmas[0][0:200])

---
Bigrams and trigrams are two- and three-word (token) strings that "belong together". Like "big_data" and "University_of_Vaasa". Gensim has functions that recongise bigrams and trigrams automatically from the corpus. Unfortunately these functions need a lot of tweaking to achieve satisfactory performance. You need to try many different parameter values for *threshold* and *min_count*.  

Below the trigrams are commented off.

bigram = gensim.models.Phrases(docs_lemmas,threshold=100, min_count=5)
#trigram = gensim.models.Phrases(bigram[docs_lemmas], threshold=1, min_count=2)  
bigram_mod = gensim.models.phrases.Phraser(bigram)
#trigram_mod = gensim.models.phrases.Phraser(trigram)

docs_bigrams = [bigram_mod[doc] for doc in docs_lemmas]

#docs_trigrams = [trigram_mod[doc] for doc in docs_bigrams]

" | ".join(docs_bigrams[0][200:300])

---
Below the words (lemmas) are connected to a unique ID. Also the extreme cases are removed from the tokens (words). A token that is in less that 3 documents, is removed. Also a token that is in more than 70 % of the documents, is removed.

id2word = corpora.Dictionary(docs_bigrams)
id2word.filter_extremes(no_below=3, no_above=0.9, keep_n=100000)

id2word.token2id['strategy']

---
The documents are changed to a bag-of-words format. That means just what it says. Documents are only a bag of words, where the order of the words is not important. We only save how many times each word is present in a document.

corpus = [id2word.doc2bow(text) for text in docs_bigrams]

---
An example of bag-of-words. The first document has four times the word "ability" that has the id "0", the word "able" that has the id "1" etc.

id2word.doc2bow(docs_bigrams[0])[0:10]

id2word[0],id2word[1]

---
Here we do the LDA modelling. We only need to decide the number of topics. LDA assumes that documents are build from these topics with varying topic importance between topics. Furthermore, the topics are assumed to probability distributions over words in our dictionary. So, topics differ from each other by their word importances (probabilities).

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=7, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=60,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True,
                                            eta = 'auto')

lda_model.print_topics()

---
pyLDAvis is used to visualise the results.

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)

vis

---
The for-loop below is used to form a dataframe with topics as columns and the most important words of each topic as rows.

top_words_df = pd.DataFrame()
for i in range(7):
    temp_words = lda_model.show_topic(i,10)
    just_words = [name for (name,_) in temp_words]
    top_words_df['Topic ' + str(i+1)] = just_words

top_words_df

---
Below I collect the importance of the topics for each document and also the publication year of each document. Using these, I can plot a time series that shows the evolution of topics in time.

evolution = np.zeros([198,7])
ind = 0
for bow in corpus:
    topics = lda_model.get_document_topics(bow)
    for topic in topics:
        evolution[ind,topic[0]] = topic[1]
    ind+=1

evolution_df = pd.DataFrame(evolution,columns = ['Topic 1','Topic 2','Topic 3','Topic 4','Topic 5','Topic 6','Topic 7'])
evolution_df['Year'] = file_years
evolution_df['Date'] = pd.to_datetime(evolution_df['Year'])
evolution_df.set_index('Date',inplace=True)
evolution_df.drop('Year',axis=1,inplace = True)

plt.style.use('bmh')
plt.rcParams["figure.figsize"] = (10, 6)
evolution_df.groupby('Date').mean().rolling(4,min_periods=1).mean().plot()

fig, axs = plt.subplots(3,2,figsize=(10,10),squeeze=True)
ind = 0
for ax in axs.flat:
    ax.scatter(evolution_df.index,evolution_df['Topic ' + str(ind+1)].rolling(16,min_periods = 1).mean(),s=1)
    ax.set_title('Topic '+str(ind+1))
    ax.set_ylim([0,1])
    ind+=1
plt.subplots_adjust(hspace=0.3)
plt.savefig('scatter_plots.png')

top_words_df = pd.DataFrame()
for i in range(12):
    top_words_df['Topic ' + str(i+1)] = [item[0] for item in lda_model.show_topic(i,20)]

---
In the LDA modelling we only decide the number of topics present in the documents and the algorithm does the rest. However, the optimal number of topcis is difficult to find. There are many different metrics, that try to measure this "optimality". These coherence-measures, however, are far from perfect. The code below takes a lot of time to execute!

# Solve the optimal number of topics
coh_list = []
perp_list = []
for i in range(20):
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,num_topics=i+2,random_state=10,update_every=1,chunksize=len(corpus)/3,passes=10,alpha='auto',per_word_topics=False,eta='auto')
    coherence_model_lda = CoherenceModel(model=lda_model, texts = docs_bigrams,corpus=corpus, dictionary=id2word, coherence='u_mass')
    coh_list.append(coherence_model_lda.get_coherence())
    perp_list.append(lda_model.log_perplexity(corpus))

coh_list

perp_list

