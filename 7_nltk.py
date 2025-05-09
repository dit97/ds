#!/usr/bin/env python
# coding: utf-8

# In[2]:


import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Download NLTK resources (only run once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample document
doc = "Text analytics is the process of transforming unstructured text into meaningful data for analysis."
#If import dataset
# df = pd.read_csv('your_file.csv')
# df.dropna(subset=['text'], inplace=True)
# doc = df['text'].tolist()

# 1. Tokenization
tokens = word_tokenize(doc)
print("Tokens:", tokens)

# 2. POS Tagging
pos_tags = pos_tag(tokens)
print("POS Tags:", pos_tags)

# 3. Stop Words Removal
stop_words = set(stopwords.words('english'))
tokens_no_stop = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
print("After Stop Words Removal:", tokens_no_stop)

# 4. Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in tokens_no_stop]
print("After Stemming:", stemmed_words)

# 5. Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens_no_stop]
print("After Lemmatization:", lemmatized_words)

# ----------------------------
# 2. Document Representation
# ----------------------------

# You can use a list of documents for TF and TF-IDF
documents = [
    "Text analytics is important.",
    "Analytics involves processing data.",
    "We analyze unstructured text in text analytics."
]

# Term Frequency (TF) word frequency
vectorizer_tf = CountVectorizer()
tf_matrix = vectorizer_tf.fit_transform(documents)
print("\nTerm Frequency (TF):")
print(tf_matrix.toarray())
print("TF Vocabulary:", vectorizer_tf.get_feature_names_out())

# TF-IDF word importance
vectorizer_tfidf = TfidfVectorizer()
tfidf_matrix = vectorizer_tfidf.fit_transform(documents)
print("\nTF-IDF:")
print(tfidf_matrix.toarray())
print("TF-IDF Vocabulary:", vectorizer_tfidf.get_feature_names_out())


# Tokenization: Splitting text

# POS Tagging: Word classification (noun)

# Stopwords Removal: Removing common words (is the})

# Stemming: Word root extraction (removing ing)

# Lemmatization: Dictionary form conversion


# In[ ]:




