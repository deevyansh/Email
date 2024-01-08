import streamlit as st
import pickle
import string
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.data.path.append("/path/to/nltk_data")
from nltk.tokenize import  word_tokenize
# Now you can use the NLTK library for tokenization

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))
st.title("Email/SMS Spam Classifier")
input_sms =st.text_input("Enter the Message")


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


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
      if i not in string.punctuation:
        y.append(i)
  text=y[:]
  y.clear()
  for i in text:
    y.append(ps.stem(i))
  return " ".join(y)

if st.button('Predict'):
  # 1. preprocess
  transformed_sms=transform_text(input_sms)
  # 2. vectorize
  vector_input=tfidf.transform([transformed_sms])
  #predict
  result=model.predict(vector_input)[0]
  #Display
  if result==1:
      st.header("Spam")
  else:
      st.header("Not Spam")

