import os
import xml.etree.ElementTree as ET
import csv
import nltk
import gensim
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize 
from gensim.models import Word2Vec
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cluster import KMeans 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances

import nltk
import string
import matplotlib.pyplot as plt
from matplotlib import pyplot
# nltk.download('stopwords')
# nltk.download('punkt')
# plt.style.use('fivethirtyeight')
import os, sys, email,re
from sklearn.feature_extraction.text import TfidfVectorizer
# def loop_directory(directory: 'D:\BTP2'):
i=0
rowlist=[]
stop_words = set(stopwords.words('english'))  
stop_words.add('The')
stop_words.add('An')
chapters_titles=[]
csv_file=open('data.csv','a+')
csv_vector=open('vector.csv','a+')
csv_writer=csv.writer(csv_file)
csv_writer_vec=csv.writer(csv_vector)
vocab=[]
for filename in os.listdir('D:\BTP2\dataset'):
    if filename.endswith(".xml"):
        # print(filename)
        # print('\n')
        mytree=ET.parse(os.path.join('D:\BTP2\dataset', filename))
        myroot=mytree.getroot()
        vector=np.array([0]*100)
        for x in myroot:
            # print(x.attrib['name'])
            chapters_titles.append(x.attrib['name'])
            word_tokens = word_tokenize(x.attrib['name'])
            # filtered_sentence = ""
            filtered_sentence=[]
            for w in word_tokens:  
                if w.lower() not in stop_words:  
                    filtered_sentence.append(w.lower())
                    # vocab.append(w)
        

            vocab.append(filtered_sentence)
            # print(filtered_sentence)
            rowlist.insert(i+1,filtered_sentence)
        csv_writer.writerow([filename,rowlist])	
    rowlist=[]

# print(vocab)
model=Word2Vec(vocab,min_count=1)
# print(model)
all_vectors = []
for index, vector in enumerate(model.wv.vectors):
    vector_object = {}
    vector_object[list(model.wv.vocab.keys())[index]] = vector
    all_vectors.append(vector_object)
# print(model['relations'])
# print(all_vectors)
# print(model['continuity'])
#=======================================================================================
# X = model[model.wv.vocab]
# pca = PCA(n_components=2)
# result = pca.fit_transform(X)
# pyplot.scatter(result[:, 0], result[:, 1])
# words = list(model.wv.vocab)
# for i, word in enumerate(words):
# 	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
# pyplot.show()
            
#========================================================================================
final_map=[]
final_array=[]
for filename in os.listdir('D:\BTP2\dataset'):
    if filename.endswith(".xml"):
        mytree=ET.parse(os.path.join('D:\BTP2\dataset', filename))
        myroot=mytree.getroot()
        vector=np.array([0]*100)
        for x in myroot:
            title=x.attrib['name']
            word_tokens = word_tokenize(x.attrib['name'])
            filtered_sentence=[]
            buckets = [0] * 100
            number=0
            for w in word_tokens:
                if w.lower() in stop_words: 
                    continue 
                buckets=np.add(model[w.lower()],buckets)
                number+=1
            for i in range(100):
                buckets[i]/=number
            final_array.append(buckets)
            case={title:buckets}
            final_map.append(case)
#=============================================================================
# print(final_map)
# X = final_array
# pca = PCA(n_components=2)
# result = pca.fit_transform(X)        
# pyplot.scatter(result[:, 0], result[:, 1])
# words = chapters_titles
# for i, word in enumerate(words):
# 	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
# pyplot.show()   
#===============================================================================

X = final_array
pca = PCA(n_components=2)
result = pca.fit_transform(X) 

Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(result)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()



# kmeans = KMeans(n_clusters= 10)
# label = kmeans.fit_predict(result)
# # print(chapters_titles)
# # print("\n")
# # print(label)

# for i in range(len(chapters_titles)):
#     print(chapters_titles[i]," ----> ",label[i])
#     print("\n")

# centroids = kmeans.cluster_centers_
# u_labels = np.unique(label)
# for i in u_labels:
#     plt.scatter(result[label == i , 0] , result[label == i , 1] , label = i)
# plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
# plt.legend()
# plt.show()

/*
CODED BY-:
             __              ___          __
|   | _|_   |__) /\  |      |   _   |  | |__)  _|_    /\
|___|  |__  |   /~~\ |___   |__| |  |__| |      |__  /~~\

*/
