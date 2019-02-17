# -*- coding: utf-8 -*-

#Importing libraries
import pandas as pd
import nltk
import string
from sklearn.model_selection import train_test_split   
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords') 
nltk.download('wordnet') 
from nltk.corpus import stopwords  

# Importing the dataset

amazon = pd.read_csv('1000.csv')

#Removing null entries

amazon.isnull().sum()
amazon = amazon.fillna(' ')
amazon.shape

# Text Length 

amazon['text length'] = amazon['reviewText'].apply(len)

# Creating a class with only 5 and 1 stars 

amazon_class = amazon[(amazon['overall'] <3 ) | (amazon['overall'] >3)]
amazon_class.shape

# Generating X and Y coordinates

X = amazon_class['reviewText']
y = amazon_class['overall']

# Resetting key values

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Data Preprocessing

def text_process(text):


    nopunc = [char for char in text if char not in string.punctuation]

    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# Vectorizing reviews
    
bow_transformer = CountVectorizer(analyzer=text_process).fit(X)
len(bow_transformer.vocabulary_)

X = bow_transformer.transform(X)

# Split Dataset into training and testing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) 

# Training the model

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)  
classifier.fit(X_train, y_train) 

# Predicting sentiment

y_pred = classifier.predict(X_test)  

# Calculating score

def eval_predictions(y_test, y_pred):
    print('accuracy:', metrics.accuracy_score(y_test, y_pred))
    print('precision:', metrics.precision_score(y_test, y_pred, average='weighted'))
    print('recall:', metrics.recall_score(y_test, y_pred, average='weighted'))
    print('F-measure:', metrics.f1_score(y_test, y_pred, average='weighted'))
eval_predictions(y_test, y_pred)