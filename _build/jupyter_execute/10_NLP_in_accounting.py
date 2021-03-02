## Introduction to natural language processing

Natural language processing (NLP) is a collective term referring to computational processing of human languages. It includes methods that analyse human-produced text, and methods that create natural language as output. Compared to many other machine learning tasks, natural language processing is very challenging, as human language is inherently ambiguous, ever-changing, and not well-defined. 

![read_robot](./images/read_robot.jpg)

There is a need for better and better NLP-algorithms, as information in the textual format is increasing exponentially.

import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(10,6))
plt.plot(np.linspace(1,10,50),np.exp(np.linspace(1,10,50)))
plt.xticks(range(1,11),labels=range(2010,2020))
plt.xlabel('Year')
plt.ylabel('Data')
plt.show()

Until 2014, core NLP techniques were dominated by linear modelling approaches that use supervised learning. Key algorithms were simple neural networks, support vector machines and logistic regression, trained over high dimensional and sparse feature vectors (bag-of-words -vectors).

![SVM](./images/svm.png)


### Topic models
A topic model is a type of statistical model for inferring the "topics" or "themes" that occur in a collection of documents. Topic modelling is a popular tool for the discovery of hidden semantic structures in a text body. Topic models assume that there are typical words that appear more frequently in a document with a certain topic. Moreover, some words are especially rare for a certain topic and for some words, there is no difference between a document with the topic and other documents. The "topics" produced by topic modelling techniques are clusters of similar words. For example, a very popular topic model called Latent Dirichlet Allocation assumes that documents are distributions of topics and topics are distributions of words.

![topic_model](./images/topic_model.gif)



### NLP example - LDA and other summarisation tools
In this example, we analyse a collection of academic journals with Latent Dirichlet Allocation (LDA) and other summarisation tools.

To manipulate and list files in directories, we need the **os** -library.

import os

I have the data in the acc_journals -folder that is under the work folder. Unfortunately, this data is not available anywhere. If you want to follow this example, just put a collection of txt-files to a "acc_journals"-folder (under the work folder) and follow the steps.

data_path = './acc_journals/'

**listdir()** makes a list of filenames inside **data_path**

files = os.listdir(data_path)

The name of the first text file is "2019_1167.txt".

files[0]

In total, we have 2126 articles.

len(files)

The filenames have a publication year as the first four digits of the name. With the following code we can collect the publication years to a list.

file_years = []
for file in files:
    file_years.append(int(file[:4])) # Pick the first four letters from the filename -string and turn it into a integer.

[file_years.count(a) for a in set(file_years)]

In this example we will need **numpy** to manipulate arrays, and **Matplotlib** for plots.

import numpy as np
import matplotlib.pyplot as plt

Let's plot the number of documents per year.

plt.style.use('fivethirtyeight') # Define the style of figures.
plt.figure(figsize=[10,6]) # Define the size of figures.
# The first argument: years, the second argument: the list of document frequencies for different years (built using list comprehension)
plt.bar(list(set(file_years)),[file_years.count(a) for a in set(file_years)]) 
plt.xticks(list(set(file_years))) # The years below the bars
plt.show()

The common practice is to remove **stopwords** from documents. These are common fill words that do not contain information about the content of documents. We use the stopwords-list from the NLTK library (www.nltk.org). Here is a short description of NLTK from their web page: "NLTK is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries, and an active discussion forum."

from nltk.corpus import stopwords

stop_words = stopwords.words("english")

Here are some example stopwords.

stop_words[0:10]

Usually, stopword-lists are extended with useless words specific to the corpus we are analysing. That is done below. These additional words are found by analysing the results at different steps of the analysis. Then the analysis is repeated.

stop_words.extend(['fulltext','document','downloaded','download','emerald','emeraldinsight',
                   'accepted','com','received','revised','archive','journal','available','current',
                   'issue','full','text','https','doi','org','www','com''ieee','cid','et','al','pp',
                   'vol','fig','reproduction','prohibited','reproduced','permission','accounting','figure','chapter'])

The following code reads every file in the **files** list and reads it content as an item to the **raw_text** list. Thus, we have a list with 2126 items and where each item is a raw text of one document.

raw_text = []
for file in files:
    fd = open(os.path.join(data_path,file),'r',errors='ignore')
    raw_text.append(fd.read())

Here is an example of raw text from the first document.

raw_text[0][3200:3500]

For the following steps, we need Gensim that is a multipurpose NLP library specially designed for topic modelling.

Here is a short description from the Gensim Github-page ([github.com/RaRe-Technologies/gensim](https://github.com/RaRe-Technologies/gensim)):

"Gensim is a Python library for topic modelling, document indexing and similarity retrieval with large corpora. Target audience is the natural language processing (NLP) and information retrieval (IR) community.
Features:
* All algorithms are memory-independent w.r.t. the corpus size (can process input larger than RAM, streamed, out-of-core),
* Intuitive interfaces
    * Easy to plug in your own input corpus/datastream (trivial streaming API)
    * Easy to extend with other Vector Space algorithms (trivial transformation API)
* Efficient multicore implementations of popular algorithms, such as online Latent Semantic Analysis (LSA/LSI/SVD), Latent Dirichlet Allocation (LDA), Random Projections (RP), Hierarchical Dirichlet Process (HDP) or word2vec deep learning.
* Distributed computing: can run Latent Semantic Analysis and Latent Dirichlet Allocation on a cluster of computers."

import gensim

Gensim has a convenient **simple_preprocess()** -function that makes many text cleaning procedures automatically. It converts a document into a list of lowercase tokens, ignoring tokens that are too short (less than two characters) or too long (more than 15 characters). The following code goes through all the raw texts and applies **simple_preprocess()** to them. So, docs_cleaned is a list with lists of tokens as items.

docs_cleaned = []
for item in raw_text:
    tokens = gensim.utils.simple_preprocess(item)
    docs_cleaned.append(tokens)

Here is an example from the first document after cleaning. The documents are now lists of tokens, and here the list is joined back as a string of text.

" ".join(docs_cleaned[0][300:400])

Next, we remove the stopwords from the documents. 

docs_nostops = []
for item in docs_cleaned:
    red_tokens = [word for word in item if word not in stop_words]
    docs_nostops.append(red_tokens)

An example text from the first document after removing the stopwords.

" ".join(docs_nostops[0][300:400])

As a next step, we remove everything else but nouns, adjectives, verbs and adverbs recognised by our language model. As our language model, we use the large English model from Spacy ([spacy.io](https://spacy.io/)). Here are some key features of Spacy from their web page:
* Non-destructive tokenisation
* Named entity recognition
* Support for 59+ languages
* 46 statistical models for 16 languages
* Pretrained word vectors
* State-of-the-art speed
* Easy deep learning integration
* Part-of-speech tagging
* Labelled dependency parsing
* Syntax-driven sentence segmentation
* Built-in visualisers for syntax and NER
* Convenient string-to-hash mapping
* Export to NumPy data arrays
* Efficient binary serialisation
* Easy model packaging and deployment
* Robust, rigorously evaluated accuracy

First, we load the library.

import spacy

Then we define that only certain part-of-speed (POS) -tags are allowed.

allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']

The **en_core_web_lg** model is quite large (700MB) so it takes a while to download it. We do not need the dependency parser  or named-entity-recognition, so we disable them.

nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])

The following code goes through the documents and removes words that are not recognised by our language model as nouns, adjectives, verbs or adverbs.

docs_lemmas = []
for red_tokens in docs_nostops:
    doc = nlp(" ".join(red_tokens)) # We need to join the list of tokens back to a single string.
    docs_lemmas.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

Here is again an example text from the first document. Things are looking good. We have a clean collection of words that are meaningful when we try to figure out what is discussed in the text.

" ".join(docs_lemmas[0][300:400])

Next, we build our bigram-model. Bigrams are two-word pairs that naturally belong together. Like the words New and York. We connect these words before we do the LDA analysis.

bigram = gensim.models.Phrases(docs_lemmas,threshold = 80, min_count=3) 
bigram_mod = gensim.models.phrases.Phraser(bigram)

docs_bigrams = [bigram_mod[doc] for doc in docs_lemmas]

In this sample text, the model creates bigrams academic_library, starting_point and bitcoin_seizure. The formed bigrams are quite good. It is very often difficult to set the parameters of **Phrases** so that we have only reasonable bigrams.

" ".join(docs_bigrams[0][300:400])

### LDA

![lda](./images/topic_model.gif)

Okay, let's start our LDA analysis. For this, we need the **corpora** module from Gensim.

import gensim.corpora as corpora

First, with, **Dictionary**, we build our dictionary using the list docs_bigrams. Then we filter the most extreme cases from the dictionary:
* no_below = 2 : no words that are in less than two documents
* no_above = 0.7 : no words that are in more than 70 % of the documents
* keep_n = 50000 : keep the 50000 most frequent words

id2word = corpora.Dictionary(docs_bigrams)
id2word.filter_extremes(no_below=2, no_above=0.7, keep_n=50000)

Then, we build our corpus by indexing the words of documents using our **id2word** -dictionary

corpus = [id2word.doc2bow(text) for text in docs_bigrams]

**corpus** contains a list of tuples for every document, where the first item of the tuples is a word-index and the second is the frequency of that word in the document.

For example, in the first document, a word with index 0 in our dictionary is once, a word with index 1 is four times, etc.

corpus[0][0:10]

We can check what those words are just by indexing our dictionary **id2word**. As you can see, 'accept' is 50 times in the first document.

[id2word[i] for i in range(10)]

This is the main step in our analysis; we build the LDA model. There are many parameters in Gensim's **LdaModel**. Luckily the default parameters work quite well. You can read more about the parameters from [radimrehurek.com/gensim/models/ldamodel.html](https://radimrehurek.com/gensim/models/ldamodel.html)

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=10, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=len(corpus),
                                           passes=10,
                                           alpha='asymmetric',
                                           per_word_topics=False,
                                            eta = 'auto')

**pyLDAvis** is a useful library to visualise the results: [github.com/bmabey/pyLDAvis](https://github.com/bmabey/pyLDAvis).

import pyLDAvis.gensim

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)

With pyLDAvis, we get an interactive figure with intertopic distances and the most important words for each topic. More separate the topic "bubbles" are, better the model.

vis

We need **pandas** to present the most important words in a dataframe.

import pandas as pd

The following code builds a dataframe from the ten most important words for each topic. Now our task would be to figure out the topics from these words.

top_words_df = pd.DataFrame()
for i in range(10):
    temp_words = lda_model.show_topic(i,10)
    just_words = [name for (name,_) in temp_words]
    top_words_df['Topic ' + str(i+1)] = just_words

top_words_df

The following steps will build a figure with the evolution of each topic. These are calculated by evaluating the weight of each topic in the documents for a certain year and then summing up these weights.

evolution = np.zeros([len(corpus),10]) # We pre-build the numpy array filled with zeroes.
ind = 0
for bow in corpus:
    topics = lda_model.get_document_topics(bow)
    for topic in topics:
        evolution[ind,topic[0]] = topic[1]
    ind+=1

We create a pandas dataframe from the NumPy array and add the years and column names.

evolution_df = pd.DataFrame(evolution)
evolution_df['Year'] = file_years
evolution_df['Date'] = pd.to_datetime(evolution_df['Year'],format = "%Y") # Change Year to datetime-object.
evolution_df.set_index('Date',inplace=True) # Set Date as an index of the dataframe
evolution_df.drop('Year',axis=1,inplace = True)

plt.style.use('fivethirtyeight')
fig,axs = plt.subplots(5,2,figsize = [15,10])
for ax,column in zip(axs.flat,evolution_df.groupby('Date').mean().columns):
    ax.plot(evolution_df.groupby('Date').mean()[column])
    ax.set_title(column,{'fontsize':14})
plt.subplots_adjust(hspace=0.4)

Next, we plot the marginal topic distribution, i.e., the relative importance of the topics. It is calculated by summing the topic weights of all documents.

First, we calculate the topic weights for every document.

doc_tops = []
for doc in corpus:
    doc_tops.append([item for (_,item) in lda_model.get_document_topics(doc)])
doc_tops_df = pd.DataFrame(doc_tops,columns=top_words_df.columns)

Then, we sum (and plot) these weights.

doc_tops_df = doc_tops_df/doc_tops_df.sum().sum()

doc_tops_df.sum(axis=0).plot.bar()

As our subsequent analysis, let's search the most representative document for each topic. It is the document that has the largest weight for a certain topic.

doc_topics = []
for doc in corpus:
    doc_topics.append([item for (_,item) in lda_model.get_document_topics(doc)])               

temp_df = pd.DataFrame(doc_topics)

With **idxmax()**, we can pick up the index that has the largest weight.

temp_df.idxmax()

We can now use **files** to connect indices to documents. For example, the most representative document of Topic 1 (index 0) is "PRACTICING SAFE COMPUTING: A MULTIMETHOD EMPIRICAL EXAMINATION OF HOME COMPUTER USER SECURITY BEHAVIORAL INTENTIONS"

files[1172]

raw_text[1172][0:500]

Let's build a master table that has the document names and other information.

master_df = pd.DataFrame({'Article':files})

First, we add the most important topic of each article to the table.

top_topic = []
for doc in corpus:
    test=lda_model.get_document_topics(doc)
    test2 = [item[1] for item in test]
    top_topic.append(test2.index(max(test2))+1)

master_df['Top topic'] = top_topic

With **head()**, we can check the first (ten) values of our dataframe.

master_df.head(10)

**value_counts()** for the "Top topic" -column can be used to check that in how many documents each topic is the most important.

master_df['Top topic'].value_counts()

### Summarisation

Let's do something else. Gensim also includes efficient summarisation-functions (these are not related to LDA any more):

From the **summarization** -module, we can use **summarize** to automatically build a short summarisation of the document. Notice that we use the original documents for this and not the preprocessed ones. **ratio = 0.01** means that the length of the summarisation should be 1 % from the original document.

summary = []
for file in raw_text:
    summary.append(gensim.summarization.summarize(file.replace('\n',' '),ratio=0.01))

Below is an example summary for the first document.

summary[1]

master_df['Summaries'] = summary

master_df

The same **summarization** -module also includes a function to search keywords from the documents. It works in the same way as **summarize()**.

keywords = []
for file in raw_text:
    keywords.append(gensim.summarization.keywords(file.replace('\n',' '),ratio=0.01).replace('\n',' '))

Here are keywords for the second document

keywords[1]

master_df['Keywords'] = keywords

master_df

### Similarities
As a next example, we analyse the similarities between the documents.

Gensim has a specific function for that: **SparseMatrixSimilarity**

from gensim.similarities import SparseMatrixSimilarity

The parameters to the function are the word-frequency corpus and the length of the dictionary.

index = SparseMatrixSimilarity(corpus,num_features=len(id2word))

sim_matrix = index[corpus]

In the similarity matrix, the diagonal has values 1 (similarity of a document with itself). We replace those values with zero to find the most similar documents from the corpus.

for i in range(len(sim_matrix)):
    sim_matrix[i,i] = 0

**Seaborn** (https://seaborn.pydata.org/) has a convenient heatmap function to plot similarities. Here is information about Seaborn from their web page: Seaborn is a Python data visualization library based on Matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.

sns.heatmap(sim_matrix)

We search the most similar article by locating the index with a largest value for every column.

master_df['Most_similar_article'] = np.argmax(sim_matrix,axis=1)

master_df

### Cluster model

Next, we build a topic analysis using a different approach. We create a TF-IDF model from the corpus and apply K-means clustering for that model.

Explanation of TF-IDF from Wikipedia: "In information retrieval, TF–IDF or TFIDF, short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. It is often used as a weighting factor in searches of information retrieval, text mining, and user modelling. The TF–IDF value increases proportionally to the number of times a word appears in the document and is offset by the number of documents in the corpus that contain the word, which helps to adjust for the fact that some words appear more frequently in general."

Explanation of K-means clustering from Wikipedia: "k-means clustering is a method of vector quantization, originally from signal processing, that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean (cluster centres or cluster centroid), serving as a prototype of the cluster...k-means clustering minimizes within-cluster variances (squared Euclidean distances)..."

![kmeans](./images/kmeans.gif)

From **Gensim.models** we pick up **TfidModel**.

from gensim.models import TfidfModel

As parameters, we need the corpus and the dictionary.

tf_idf_model = TfidfModel(corpus,id2word=id2word)

We use the model to build up a TF-IDF -transformed corpus.

tform_corpus = tf_idf_model[corpus]

**corpus2csc** converts a streamed corpus in bag-of-words format into a sparse matrix, with documents as columns.

spar_matr = gensim.matutils.corpus2csc(tform_corpus)

spar_matr

Sparse matrix to normal array. Also, we need to transpose it for the K-means model.

tfidf_matrix = spar_matr.toarray().transpose()

tfidf_matrix

tfidf_matrix.shape

Scikit-learn has a function to form a K-means clustering model from a matrix. It is done below. We use ten clusters.

from sklearn.cluster import KMeans

kmodel = KMeans(n_clusters=10)

kmodel.fit(tfidf_matrix)

clusters = km.labels_.tolist()

master_df['Tf_idf_clusters'] = clusters

km.cluster_centers_

master_df

Let's collect the ten most important words for each cluster.

centroids = km.cluster_centers_.argsort()[:, ::-1] # Sort the words according to their importance.

for i in range(num_clusters):
    j=i+1
    print("Cluster %d words:" % j, end='')
    for ind in centroids[i, :10]:
        print(' %s' % id2word.id2token[ind],end=',')
    print()
    print()

With a smaller corpus, we could use fancy visualisations, like multidimensional scaling and Ward-clustering, to represent the document relationship. However, with over 2000 documents, that is not meaningful. Below are examples of both (not related to our analysis).

![mds](./images/mds.png)

![ward](./images/ward.jpg)

With pandas **crosstab**, we can easily check the connection between the LDA topics and the K-means clusters.

pd.crosstab(master_df['Top topic'],master_df['Tf_idf_clusters'])

