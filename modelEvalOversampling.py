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
import time
import timeit

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
  
ros = RandomOverSampler(random_state=0)
resampled_X, resampled_yhs = ros.fit_resample(X,y_hs)
print(Counter(resampled_yhs))

data_trainhs = []
#preprocess
factory = StemmerFactory()
stopword_factory = StopWordRemoverFactory()
stemmer = factory.create_stemmer()
stopword = stopword_factory.create_stop_word_remover()
for i in resampled_X:
    stem = stemmer.stem(i[0])
    stop = stopword.remove(stem)
    data_trainhs.append(stop)
    
tfidfhs = TfidfVectorizer()
tfidfhs.fit(data_trainhs)

#KNN
#UJI COBA OVERSAMPLING
ros = RandomOverSampler(random_state=0)
resampled_X, resampled_konteks = ros.fit_resample(X,y_konteks)
print(Counter(resampled_konteks))
data_trainkonteks = []
# # #preprocess
factory = StemmerFactory()
stopword_factory = StopWordRemoverFactory()
stemmer = factory.create_stemmer()
stopword = stopword_factory.create_stop_word_remover()
for i in resampled_X:
    stem = stemmer.stem(i[0])
    stop = stopword.remove(stem)
    data_trainkonteks.append(stop)
tfidfkonteks = TfidfVectorizer()
tfidfkonteks.fit(data_trainkonteks)

naive_bayes = MultinomialNB()
rf = RandomForestClassifier()
svm_classifier = svm.SVC(kernel='linear')
khs = round(math.sqrt(len(data_trainhs)))
knnHs = KNeighborsClassifier(n_neighbors=khs, metric="euclidean")
kKonteks = round(math.sqrt(len(data_trainkonteks)))
knnKonteks = KNeighborsClassifier(n_neighbors=kKonteks, metric="euclidean")

skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=1)
nb_hs = []
nb_konteks = []
knn_hs = []
knn_konteks = []
rf_hs = []
rf_konteks = []
svm_hs = []
svm_konteks = []

# Hate speech / non hate speech
for train_index, test_index in skf.split(data_trainhs, resampled_yhs):
    x_train_fold, x_test_fold = np.array(data_trainhs).reshape(-1,1)[train_index],np.array(data_trainhs).reshape(-1,1)[test_index]
    y_train_fold, y_test_fold = np.array(resampled_yhs)[train_index], np.array(resampled_yhs)[test_index]
    tfidf_train = tfidfhs.transform(x_train_fold.ravel())
    tfidf_test = tfidfhs.transform(x_test_fold.ravel())
    start_timenbhs = timeit.default_timer()
    naive_bayes.fit(tfidf_train,y_train_fold.ravel())
    nb_hs.append(naive_bayes.score(tfidf_test, y_test_fold.ravel()))
    end_timenbhs = timeit.default_timer()
    ex_timenbhs = end_timenbhs-start_timenbhs
    start_timeknnhs = timeit.default_timer()
    knnHs.fit(tfidf_train,y_train_fold.ravel())
    knn_hs.append(knnHs.score(tfidf_test, y_test_fold.ravel()))
    end_timeknnhs = timeit.default_timer()
    ex_timeknnhs = end_timeknnhs-start_timeknnhs
    start_timerfhs = timeit.default_timer()
    rf.fit(tfidf_train,y_train_fold.ravel())
    rf_hs.append(rf.score(tfidf_test, y_test_fold.ravel()))
    end_timerfhs = timeit.default_timer()
    ex_timerfhs = end_timerfhs-start_timerfhs
    start_timesvmhs = timeit.default_timer()
    svm_classifier.fit(tfidf_train,y_train_fold.ravel())
    svm_hs.append(svm_classifier.score(tfidf_test, y_test_fold.ravel()))
    end_timesvmhs = timeit.default_timer()
    ex_timesvmhs = end_timesvmhs-start_timesvmhs
    
for train_index, test_index in skf.split(data_trainkonteks, resampled_konteks):
    xk_train_fold, xk_test_fold = np.array(data_trainkonteks).reshape(-1,1)[train_index],np.array(data_trainkonteks).reshape(-1,1)[test_index]
    yk_train_fold, yk_test_fold = np.array(resampled_konteks)[train_index], np.array(resampled_konteks)[test_index]
    tfidf_traink = tfidfkonteks.transform(xk_train_fold.ravel())
    tfidf_testk = tfidfkonteks.transform(xk_test_fold.ravel())
    start_timenbk = timeit.default_timer()
    naive_bayes.fit(tfidf_traink,yk_train_fold.ravel())
    nb_konteks.append(naive_bayes.score(tfidf_testk, yk_test_fold.ravel()))
    end_timenbk = timeit.default_timer()
    ex_timenbk = end_timenbk-start_timenbk
    start_timeknnk = timeit.default_timer()
    knnKonteks.fit(tfidf_traink,yk_train_fold.ravel())
    knn_konteks.append(knnKonteks.score(tfidf_testk, yk_test_fold.ravel()))
    end_timeknnk = timeit.default_timer()
    ex_timeknnk = end_timeknnk-start_timeknnk
    start_timerfk = timeit.default_timer()
    rf.fit(tfidf_traink,yk_train_fold.ravel())
    rf_konteks.append(rf.score(tfidf_testk, yk_test_fold.ravel()))
    end_timerfk = timeit.default_timer()
    ex_timerfk = end_timerfk-start_timerfk
    start_timesvmk = timeit.default_timer()
    svm_classifier.fit(tfidf_traink,yk_train_fold.ravel())
    svm_konteks.append(svm_classifier.score(tfidf_testk, yk_test_fold.ravel()))
    end_timesvmk = timeit.default_timer()
    ex_timesvmk = end_timesvmk-start_timesvmk
    
print("\nNaive Bayes Hate Speech")
print('List of possible accuracy:', nb_hs)
print('Maximum Accuracy That can be obtained from this model is:',
      max(nb_hs)*100, '%')
print('Minimum Accuracy:',
      min(nb_hs)*100, '%')
print('Overall Accuracy:',
      mean(nb_hs)*100, '%')
print('Standard Deviation is:', stdev(nb_hs))
print('Time: ', ex_timenbhs)

print("\nNaive Bayes Konteks")
print('List of possible accuracy:', nb_konteks)
print('Maximum Accuracy That can be obtained from this model is:',
      max(nb_konteks)*100, '%')
print('Minimum Accuracy:',
      min(nb_konteks)*100, '%')
print('Overall Accuracy:',
      mean(nb_konteks)*100, '%')
print('Standard Deviation is:', stdev(nb_konteks))
print('Time: ', ex_timenbk)

print("\nKNN Hate Speech")
print('List of possible accuracy:', knn_hs)
print('Maximum Accuracy That can be obtained from this model is:',
      max(knn_hs)*100, '%')
print('Minimum Accuracy:',
      min(knn_hs)*100, '%')
print('Overall Accuracy:',
      mean(knn_hs)*100, '%')
print('Standard Deviation is:', stdev(knn_hs))
print('Time: ', ex_timeknnhs)

print("\nKNN Konteks")
print('List of possible accuracy:', knn_konteks)
print('Maximum Accuracy That can be obtained from this model is:',
      max(knn_konteks)*100, '%')
print('Minimum Accuracy:',
      min(knn_konteks)*100, '%')
print('Overall Accuracy:',
      mean(knn_konteks)*100, '%')
print('Standard Deviation is:', stdev(knn_konteks))
print('Time: ', ex_timeknnk)

print("\nRandom Forest Hate Speech")
print('List of possible accuracy:', rf_hs)
print('Maximum Accuracy That can be obtained from this model is:',
      max(rf_hs)*100, '%')
print('Minimum Accuracy:',
      min(rf_hs)*100, '%')
print('Overall Accuracy:',
      mean(rf_hs)*100, '%')
print('Standard Deviation is:', stdev(rf_hs))
print('Time: ', ex_timerfhs)

print("\nRandom Forest Konteks")
print('List of possible accuracy:', rf_konteks)
print('Maximum Accuracy That can be obtained from this model is:',
      max(rf_konteks)*100, '%')
print('Minimum Accuracy:',
      min(rf_konteks)*100, '%')
print('Overall Accuracy:',
      mean(rf_konteks)*100, '%')
print('Standard Deviation is:', stdev(rf_konteks))
print('Time: ', ex_timerfk)

print("\nSVM Hate Speech")
print('List of possible accuracy:', svm_hs)
print('Maximum Accuracy That can be obtained from this model is:',
      max(svm_hs)*100, '%')
print('Minimum Accuracy:',
      min(svm_hs)*100, '%')
print('Overall Accuracy:',
      mean(svm_hs)*100, '%')
print('Standard Deviation is:', stdev(svm_hs))
print('Time: ', ex_timesvmhs)

print("\nSVM Konteks")
print('List of possible accuracy:', svm_konteks)
print('Maximum Accuracy That can be obtained from this model is:',
      max(svm_konteks)*100, '%')
print('Minimum Accuracy:',
      min(svm_konteks)*100, '%')
print('Overall Accuracy:',
      mean(svm_konteks)*100, '%')
print('Standard Deviation is:', stdev(svm_konteks))
print('Time: ', ex_timesvmk)