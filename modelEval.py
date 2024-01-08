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
# df = pd.read_csv('DatasetCSV.csv')

# y_konteks = df.values[:,-1]

#UJI COBA OVERSAMPLING
# ros = RandomOverSampler(random_state=0)
# resampled_X, resampled_yhs = ros.fit_resample(X,y_hs)
# print(Counter(resampled_yhs))

# data_train = []
# #preprocess
# factory = StemmerFactory()
# stopword_factory = StopWordRemoverFactory()
# stemmer = factory.create_stemmer()
# stopword = stopword_factory.create_stop_word_remover()
# for i in resampled_X:
#     stem = stemmer.stem(i[0])
#     stop = stopword.remove(stem)
#     data_train.append(stop)

# tf = CountVectorizer(max_features=1000)
# TF_vector = tf.fit_transform(temp)
# # print(TF_vector)

# # print(data_train)
# tfidf = TfidfVectorizer()
# tfidf.fit(data_train)
# X_train, X_test, y_train, y_test = train_test_split(data_train, resampled_yhs, test_size = 0.2, random_state = 10)
# # # # print(X_test)
# # # # print(y_test)
# tfidf_train = tfidf.transform(X_train)
# tfidf_test = tfidf.transform(X_test)
# # # print(tfidf_test.shape)
# # # print(tfidf_train.shape)
# #Prediction secara bersamaan
# #NAIVE BAYES
naive_bayes = MultinomialNB()
# # naive_bayes.fit(tfidf_train,y_train)
# # y_pred= naive_bayes.predict(tfidf_test)
rf = RandomForestClassifier()
# # rf.fit(tfidf_train,y_train)
# # y_predrf = rf.predict(tfidf_test)
svm_classifier = svm.SVC(kernel='linear')
# svm_classifier.fit(tfidf_train,y_train)
# y_predsvm = svm_classifier.predict(tfidf_test)

# print(y_test)
# print(y_pred)
#Evaluation
# print("NAIVE BAYES")
# precision = precision_score(y_test, y_pred, average='weighted', pos_label="hate speech")
# matrix = confusion_matrix(y_test, y_pred)
# #CONFUSION MATRIX NAIVE BAYES
# sns.heatmap(matrix, 
#             annot=True,
#             fmt='g', 
#             xticklabels=['hate speech','non hate speech'],
#             yticklabels=['hate speech','non hate speech'])
# plt.ylabel('Prediction',fontsize=13)
# plt.xlabel('Actual',fontsize=13)
# plt.title('Confusion Matrix',fontsize=17)
# plt.show()

# print("Precision = ",precision)
# recall = recall_score(y_test, y_pred, average='weighted', pos_label="hate speech")
# print("Recall = ",recall)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy = ",accuracy)

#KNN
#UJI COBA OVERSAMPLING
ros = RandomOverSampler(random_state=0)
resampled_X, resampled_konteks = ros.fit_resample(X,y_konteks)
print(Counter(resampled_konteks))
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
tfidf_train = tfidf.transform(X_train)
tfidf_test = tfidf.transform(X_test)
#Tes K
# k_values = [i for i in range (1,77)]
# scores = []
#Rumus K
k = round(math.sqrt(len(data_train)))
# #EUCLIDEAN
knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
score = cross_val_score(knn, tfidf_train, y_train, cv=10)
print("Metode= Euclidean, Accuracy= ",np.mean(score))
# for k in k_values:
#       # knn = KNeighborsClassifier(n_neighbors=k, p=2)
#       # knn.fit(tfidf_train, y_train)
#       # y_predKnn = knn.predict(tfidf_test)
#       # accuracy = accuracy_score(y_test, y_predKnn)
#       # scores.append(accuracy)
#       # print("K= ",k," Accuracy= ",accuracy)
#       knn = KNeighborsClassifier(n_neighbors=k)
#       score = cross_val_score(knn, tfidf_train, y_train, cv=10)
#       print("K= ",k," Accuracy= ",np.mean(score))
#       scores.append(np.mean(score))
# sns.lineplot(x = k_values, y=scores, markers='o')
# plt.xlabel("K Values")
# plt.ylabel("Accuracy Score")
# plt.show()
# print(y_test)
# print(y_predKnn)
# Evaluation
# knn = KNeighborsClassifier(n_neighbors=k, p=2)
# knn.fit(tfidf_train, y_train)
# y_predKnn = knn.predict(tfidf_test)
# print("\nKNN EUCLIDEAN")
# precision = precision_score(y_test, y_predKnn, average='macro') #macro averaging untuk multiclass
# print("Precision = ",precision)
# recall = recall_score(y_test, y_predKnn, average='macro')
# print("Recall = " ,recall)
# accuracy = accuracy_score(y_test, y_predKnn)
# print("Accuracy = ",accuracy)
# CONFUSION MATRIX KNN
# matrixKonteks = confusion_matrix(y_test, y_predKnn, labels=['Agama','Individu','Kelompok orang','Organisasi','Ras','Suku'])
# sns.heatmap(matrixKonteks, 
#             annot=True,
#             fmt='g',
#             xticklabels=['Agama','Individu','Kelompok orang','Organisasi','Ras','Suku'],
#             yticklabels=['Agama','Individu','Kelompok orang','Organisasi','Ras','Suku'])
# plt.ylabel('Prediction',fontsize=13)      
# plt.xlabel('Actual',fontsize=13)
# plt.title('Confusion Matrix',fontsize=17)
# plt.show()
# print(metrics.classification_report(y_test,y_predKnn))
#MANHATTAN
knn = KNeighborsClassifier(n_neighbors=k, metric="manhattan")
score = cross_val_score(knn, tfidf_train, y_train, cv=10)
print("Metode= Manhattan, Accuracy= ",np.mean(score))
#Minkowski
knn = KNeighborsClassifier(n_neighbors=k,  metric="minkowski")
score = cross_val_score(knn, tfidf_train, y_train, cv=10)
print("Metode= Minkowski, Accuracy= ",np.mean(score))
#Cosine
knn = KNeighborsClassifier(n_neighbors=k,  metric="cosine")
score = cross_val_score(knn, tfidf_train, y_train, cv=10)
print("Metode= Cosine, Accuracy= ",np.mean(score))
#chebyshev
# knn = KNeighborsClassifier(n_neighbors=k,  metric="chebyshev")
# score = cross_val_score(knn, tfidf_train, y_train, cv=10)
# print("Metode= chebyshev, Accuracy= ",np.mean(score))

# stratified K-Fold Cross Val
skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=1)
nb_hs = []
nb_konteks = []
knn_hs = []
knn_konteks = []
rf_hs = []
rf_konteks = []
svm_hs = []
svm_konteks = []

# # Hate speech / non hate speech
for train_index, test_index in skf.split(data_train, y_hs):
    x_train_fold, x_test_fold = np.array(data_train).reshape(-1,1)[train_index],np.array(data_train).reshape(-1,1)[test_index]
    y_train_fold, y_test_fold = np.array(y_hs)[train_index], np.array(y_hs)[test_index]
    yk_train_fold, yk_test_fold = np.array(y_konteks)[train_index], np.array(y_konteks)[test_index]
    tfidf_train = tfidf.transform(x_train_fold.ravel())
    tfidf_test = tfidf.transform(x_test_fold.ravel())
    
    start_timenbhs = timeit.default_timer()
    naive_bayes.fit(tfidf_train,y_train_fold.ravel())
    nb_hs.append(naive_bayes.score(tfidf_test, y_test_fold.ravel()))
    end_timenbhs = timeit.default_timer()
    ex_timenbhs = end_timenbhs-start_timenbhs
    
    start_timenbk = timeit.default_timer()
    naive_bayes.fit(tfidf_train,yk_train_fold.ravel())
    nb_konteks.append(naive_bayes.score(tfidf_test, yk_test_fold.ravel()))
    end_timenbk = timeit.default_timer()
    ex_timenbk = end_timenbk-start_timenbk
    
    start_timeknnhs = timeit.default_timer()
    knn.fit(tfidf_train,y_train_fold.ravel())
    knn_hs.append(knn.score(tfidf_test, y_test_fold.ravel()))
    end_timeknnhs = timeit.default_timer()
    ex_timeknnhs = end_timeknnhs-start_timeknnhs
    
    start_timeknnk = timeit.default_timer()
    knn.fit(tfidf_train,yk_train_fold.ravel())
    knn_konteks.append(knn.score(tfidf_test, yk_test_fold.ravel()))
    end_timeknnk = timeit.default_timer()
    ex_timeknnk = end_timeknnk-start_timeknnk
    
    start_timerfhs = timeit.default_timer()
    rf.fit(tfidf_train,y_train_fold.ravel())
    rf_hs.append(rf.score(tfidf_test, y_test_fold.ravel()))
    end_timerfhs = timeit.default_timer()
    ex_timerfhs = end_timerfhs-start_timerfhs
    
    start_timerfk = timeit.default_timer()
    rf.fit(tfidf_train,yk_train_fold.ravel())
    rf_konteks.append(rf.score(tfidf_test, yk_test_fold.ravel()))
    end_timerfk = timeit.default_timer()
    ex_timerfk = end_timerfk-start_timerfk
    
    start_timesvmhs = timeit.default_timer()
    svm_classifier.fit(tfidf_train,y_train_fold.ravel())
    svm_hs.append(svm_classifier.score(tfidf_test, y_test_fold.ravel()))
    end_timesvmhs = timeit.default_timer()
    ex_timesvmhs = end_timesvmhs-start_timesvmhs
    
    start_timesvmk = timeit.default_timer()
    svm_classifier.fit(tfidf_train,yk_train_fold.ravel())
    svm_konteks.append(svm_classifier.score(tfidf_test, yk_test_fold.ravel()))
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