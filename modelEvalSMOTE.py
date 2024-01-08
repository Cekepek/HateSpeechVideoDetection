import pandas as pd
import numpy as np
import math
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
import mysql.connector
from sklearn.model_selection import StratifiedKFold
from statistics import mean, stdev
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt 
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
from collections import Counter
from sklearn.model_selection import cross_val_score
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="tugas_akhir"
)

X = []
mycursor = mydb.cursor()
mycursor.execute("SELECT Speech FROM datasets")
myresult = mycursor.fetchall()
for x in myresult:
  X.append(x)
# print(X)

y_hs = []
mycursor.execute("SELECT Hatespeech FROM datasets")
myresult = mycursor.fetchall()
for y in myresult:
  y_hs.append(y)

y_konteks = []
mycursor.execute("SELECT Konteks FROM datasets")
myresult = mycursor.fetchall()
for y in myresult:
  y_konteks.append(y)
  

data_train = []
# # #preprocess
factory = StemmerFactory()
stopword_factory = StopWordRemoverFactory()
stemmer = factory.create_stemmer()
stopword = stopword_factory.create_stop_word_remover()
for i in X:
    stem = stemmer.stem(i[0])
    stop = stopword.remove(stem)
    data_train.append(stop)
tfidf = TfidfVectorizer()
tfidf.fit(data_train)
X_train, X_test, y_train, y_test = train_test_split(data_train, y_konteks, test_size = 0.2, random_state = 10)
X_train_tfidf = tfidf.transform(X_train)
X_train_tfidf = X_train_tfidf.toarray()
X_test_tfidf = tfidf.transform(X_test)
X_test_tfidf = X_test_tfidf.toarray()


tl = TomekLinks()
X_resampled_tl, y_resampled_tl = tl.fit_resample(X_train_tfidf, y_train)

smo_te = SMOTETomek(random_state=0, tomek=tl)
smote = SMOTE(sampling_strategy='minority', random_state=42)
resampled_X, resampled_yk = smote.fit_resample(X_train_tfidf,y_train)
# print(Counter(resampled_yk))

k = round(math.sqrt(len(data_train)))
# #EUCLIDEAN
# knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
# score = cross_val_score(knn, resampled_X, resampled_yk, cv=10)
# print("Accuracy= ",np.mean(score))
knn = KNeighborsClassifier(n_neighbors=k, p=2)
# pipeline = Pipeline([
#     ('sampling', smo_te),
#     ('classifier', knn)
# ])
knn.fit(resampled_X, resampled_yk)
y_predKnn = knn.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_predKnn)
print(" Accuracy= ",accuracy)