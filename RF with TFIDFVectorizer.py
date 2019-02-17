# -*- coding: utf-8 -*-

#Importing libraries
import pandas as pd
import re  
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split   
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
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

documents = []
stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):  
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    documents.append(document)


#TFIDF Vectorization
    
tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))  
X = tfidfconverter.fit_transform(documents).toarray()

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