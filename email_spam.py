# -*- coding: utf-8 -*-
"""EMAIL SPAM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bkZm1_6Fu4HS4fHM2jn7qNTcb0zbdPvR
"""

import numpy as np

import pandas as pd

from google.colab import files

uploaded= files.upload()

df = pd.read_csv('spam.csv', encoding='latin-1')

df.sample(5)

df.shape

#Data Cleaning
#EDA
#Text Processing
#Model Building
#Evaluation
#Improvement
#Website
#Deploy

"""**Data Cleaning**"""

df.info()

#drop last 3 columns
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)

df.sample(5)

#renaming the columns
df.rename(columns={'v1':'target','v2':'text'},inplace=True)

df.sample(5)

df.head()

from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()

df['target']=encoder.fit_transform(df['target'])

df.head()

#missing value
df.isnull().sum()

#check for the duplicated values
df.duplicated().sum()

#remove duplicate
df=df.drop_duplicates(keep='first')

df.duplicated().sum()

df.shape

"""**2.EDA**"""

df['target'].value_counts()

import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct="%0.2f")
plt.show()

#Data is imbalanced

pip install nltk

import nltk
nltk.download('punkt')

df['num_characters']=df['text'].apply(len)

df.head()

# num of words
df['num_words']=df['text'].apply(lambda x:len(nltk.word_tokenize(x)))

df.head()

df['num_sentences']=df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))

df.head()

df[['num_characters','num_words','num_sentences']].describe()

#ham
df[df['target'] == 0][['num_characters', 'num_words', 'num_sentences']].describe()

#spam
df[df['target'] == 1][['num_characters', 'num_words', 'num_sentences']].describe()

import seaborn as sns

plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_characters'])
sns.histplot(df[df['target']==1]['num_characters'],color='red')

plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_words'])
sns.histplot(df[df['target']==1]['num_words'],color='red')

sns.pairplot(df,hue='target')

df.corr()

sns.heatmap(df.corr(),annot=True)

"""**3.Data Preprocessing**"""

#Lower case
#Tokenization
#Removing Special Characters
#Removing Stop Words and Punctuations
#Stemming

!pip install nltk
import nltk
!pip install stopwords
import stopwords
import string
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
ps.stem('loving')

def transform_text(text):
  text=text.lower()
  text=nltk.word_tokenize(text)
  y=[]
  for i in text:
    if i.isalnum():
      y.append(i)
  text=y[:]
  y.clear()
  for i in text:
    if i not in nltk.corpus.stopwords.words('english'):
      if i not in string.punctuation:
        y.append(i)
  text=y[:]
  y.clear()
  for i in text:
    y.append(ps.stem(i))
  return " ".join(y)

transform_text('Hi How are 20% you Deevyansh?')

df['transformed_text']=df['text'].apply(transform_text)

df.head()

from wordcloud import WordCloud
wc= WordCloud(width=500,height=500,min_font_size=10,background_color='white')

spam_wc=wc.generate(df[df['target']==1]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(12,6))
plt.imshow(spam_wc)

ham_wc=wc.generate(df[df['target']==0]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(12,6))
plt.imshow(ham_wc)

spam_corpus=[]
for msg in df[df['target']==1]['transformed_text'].tolist():
  for words in msg.split():
    spam_corpus.append(words)
len(spam_corpus)

from collections import Counter
word_counts = Counter(spam_corpus)

# Create a DataFrame from the most common 30 words and their counts
df2 = pd.DataFrame(word_counts.most_common(30), columns=['Word', 'Count'])

# Create a bar plot using Seaborn
sns.barplot(x='Word', y='Count', data=df2)
plt.xticks(rotation='vertical')

ham_corpus=[]
for msg in df[df['target']==0]['transformed_text'].tolist():
  for words in msg.split():
    ham_corpus.append(words)
len(ham_corpus)

from collections import Counter
word_counts = Counter(ham_corpus)

# Create a DataFrame from the most common 30 words and their counts
df3 = pd.DataFrame(word_counts.most_common(30), columns=['Word', 'Count'])

# Create a bar plot using Seaborn
sns.barplot(x='Word', y='Count', data=df3)
plt.xticks(rotation='vertical')

df.head()

"""**4.Model Building**"""

#Text Vectorization
#Using bag of words
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv=CountVectorizer()
tfidf=TfidfVectorizer()

X=tfidf.fit_transform(df['transformed_text']).todense()

X.shape

y=df['target'].values

y

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=2)

from sklearn.naive_bayes import GaussianNB,MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score

gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()

gnb.fit(np.asarray(X_train),Y_train)
y_pred1=gnb.predict(np.asarray(X_test))
print(accuracy_score(Y_test,y_pred1))
print(confusion_matrix(Y_test,y_pred1))
print(precision_score(Y_test,y_pred1))

mnb.fit(np.asarray(X_train),Y_train)
y_pred2=mnb.predict(np.asarray(X_test))
print(accuracy_score(Y_test,y_pred2))
print(confusion_matrix(Y_test,y_pred2))
print(precision_score(Y_test,y_pred2))

bnb.fit(np.asarray(X_train),Y_train)
y_pred3=bnb.predict(np.asarray(X_test))
print(accuracy_score(Y_test,y_pred3))
print(confusion_matrix(Y_test,y_pred3))
print(precision_score(Y_test,y_pred3))

#tfidf and mnb is choosen
import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))

