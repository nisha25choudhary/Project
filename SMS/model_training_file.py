# Importing the libraries
import pandas as pd
import re
import numpy as np
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



#read dataset from csv file
dataset = pd.read_csv('spam.csv', encoding = 'windows 1257')

#delete nan values columns
dataset = dataset.drop(dataset.columns[[2, 3, 4]], axis=1)

#store data after Cleaning the texts
corpus = []

# to store diffeerent model objects
models_dict = {}


#cleaning the texts

for i in range(0, 5572):
    #remove digits and punctuation
    sms = re.sub('[^a-zA-Z]', ' ', dataset['v2'][i])
    #Convert text to lowercase
    sms = sms.lower()
    #converting all words into list with removing white space
    sms = sms.split()
    #Remove stop words
    '''
    “Stop words” are the most common words in a language like “the”, “a”, “on”, “is”, “all”.
    '''
    sms = [word for word in sms if not word in set(stopwords.words('english'))]
    #perform stemming
    '''
    Stemming is a process of reducing words to their word stem, base or root form (for example, books — book, looked — look).
    '''
    ps = PorterStemmer()
    sms = [ps.stem(word) for word in sms]

    sms = ' '.join(sms)
    #perform all process add sms into curpus list
    corpus.append(sms)


# Creating the Bag of Words model
# Also known as the vector space model
# Text to Features (Feature Engineering on text data)
#count is >1500 onlu those words are take mean create only 1500 column
cv = CountVectorizer(max_features = 1500)
features = cv.fit_transform(corpus).toarray()
models_dict["CV"] = cv
labels = dataset.iloc[:, 0].values

#labels are ndarray object of numpy module therefore apply LabelEncoding process
labelencoder = LabelEncoder()
labels = labelencoder.fit_transform(labels)
models_dict["LabelEncoder"] = labelencoder


# Splitting the dataset into the Training set and Test set
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.20, random_state = 0)


#applying  Logistic Regression on this text dataset
# Fitting  Logistic Regression to the Training set
classifier = LogisticRegression()
classifier.fit(features_train, labels_train)
# Predicting the class labels
labels_pred_lg = classifier.predict(features_test)
#accuracy score of model
score_lg = round(accuracy_score(labels_test, labels_pred_lg)*100,2)

models_dict["Logistic_Reg"] = classifier

# Save the trained model as a pickle string.
with open("static/models/trained_models.pkl", "wb") as f:
    pickle.dump(models_dict, f)
