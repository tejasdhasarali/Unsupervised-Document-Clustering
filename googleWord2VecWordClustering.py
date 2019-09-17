#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 09:48:43 2019

@author: tejashasarali
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 20:23:14 2019

@author: tejashasarali
"""

from gensim.models import KeyedVectors
import pandas as pd
import os.path
import nltk
from sklearn.cluster import DBSCAN
import numpy as np
from random import sample
from datetime import datetime

columns=["_id","pageTitle","url"]


def tokenize(columns):
    return nltk.word_tokenize(columns[0])


class DBSCAN_Word2Vec:
    def __init__(self, size, eps=0.5, min_samples=5, n_jobs=None, metric='euclidean'):
        self.model=DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs, metric=metric)
        self.size=size
        self.eps=eps
        
    def fit(self, X, y=None, sample_weight=None):
        return self
    
    def convert(self, columns):
        output=""
        for word in columns[0]:
            if word in self.wordToMaxWord:
                output+=self.wordToMaxWord[word]+" "
            else:
                output+=word+" "
        return output
    
    def createReport(self):
        for samples, fileType in ((10, 'top'), (float('inf'), 'full')):
            with open('{}_google_{}_{}_{}.txt'.format(fileType,str(self.size),str(self.eps),int(datetime.now().timestamp())),
                      'w', encoding='utf-8') as outFile:
                
                for group in sorted(self.groupToIndex, key=lambda group: len(self.groupToIndex[group]), reverse=True):
                    if(group != -1):
                        outFile.write('Group {} - Size {} \n\n'.format(group, len(self.groupToIndex[group])))
                        docs = sample(self.groupToIndex[group], min(len(self.groupToIndex[group]), samples))
                        outFile.write('Max Word   : ' +self.wordToMaxWord[self.indexToWord[docs[0]]] + '\n')
                        outFile.write('Words: ')
                        for doc in docs:
                            outFile.write(self.indexToWord[doc]+" | ")
                        outFile.write('\n')
                        outFile.write('=' * 151 + '\n')
    
    def process(self, dbscan, vocab, word2vec):
        index=0
        self.groupToCount={}
        self.groupToIndex={}
        self.indexToWord={}
        for group, word in zip(dbscan, vocab):
            if group not in self.groupToCount:
                self.groupToCount[group]=[]
            if group not in self.groupToIndex:
                self.groupToIndex[group]=[]
            self.indexToWord[index]=word
            self.groupToCount[group].append(word2vec.vocab[word].count)
            self.groupToIndex[group].append(index)
            index+=1
            #print(groupToCount)
 
        self.wordToMaxWord={}
        for group, word in zip(dbscan, vocab):
            if(group !=-1):
                self.wordToMaxWord[word]=self.indexToWord[self.groupToIndex[group][np.argmax(self.groupToCount[group])]]
        #print(self.wordToMaxWord)   
        
    def save(self,dbscan):
        np.save("dbscan_google"+str(self.size)+"_"+str(self.eps),np.array(dbscan))
        
        


    def transform(self, vectors, y=None, sample_weight=None):
        
        dbscan=self.model.fit_predict(vectors)
        print("DBSCAN model trained")
        
        self.save(dbscan)
        print("DBSCAN model saved")
        
        return
    
    def fit_transform(self, vectors, y=None, sample_weight=None):
        
        dbscan=self.model.fit_predict(vectors)
        print("DBSCAN model trained")
        
        self.save(dbscan)
        print("DBSCAN model saved")
        
        return
    
    def createNewData(self, df, vocab, dbscan, word2Vec):
        
        self.process(dbscan, vocab, word2Vec)
        
        df['relation']= df[['relationTokens']].apply(self.convert,axis=1)
        print("Output of DBCSAN finished processing")
        
        self.createReport()
        print("Report Generated")
        
        return df
    
    
def createDataForDBSCAN(word2Vec, df):
    vectors=[]
    vocabSet=set()
    vocab =[]
    for index, row in df.iterrows():
        for token in row['relationTokens']:
            if token.isalnum() and token not in vocabSet and token in word2Vec.vocab:
                vectors.append(word2Vec[token])
                vocab.append(token)
                vocabSet.add(token)
    print("Data creation for DBSCAN completed the new length is",len(vectors))
    return (vectors, vocab)
    

print("Reading the Data for processing")
    
dataComplete=pd.DataFrame(columns=columns)

   
for i in range(5):
    if(os.path.isfile("DataProcessed "+str(i)+".json")):
        data=pd.read_json("DataProcessed "+str(i)+".json")
        if 'relationTokens' not in data.columns:
            data['relationTokens']=data[['relation']].apply(tokenize,axis=1)
        if 'relation' in data.columns:
            data=data.drop('relation', axis=1)
        dataComplete=dataComplete.append(data,ignore_index=True, sort=False)
        print("DataProcessed",i,"read")
        
print("Reading of the Data Completed")

vocab=set()
       
word2Vec = KeyedVectors.load_word2vec_format('/Users/tejashasarali/MyFiles/FSU/Spring 2019/Big Data/Project/Models/GoogleNews-vectors-negative300.bin', binary=True) 
size=word2Vec.vector_size
print("Google Word2Vec model of size",size,"loaded")    

if not (vocab):
    vectors, vocab = createDataForDBSCAN(word2Vec, dataComplete)

for eps in ([0.1, 0.25, 0.50, 0.75]):
    n_jobs = 1
    metric='cosine'
    print("Starting to train the DBSCAN model with eps=",eps,"size=",size,"n_jobs=",n_jobs,"metric",metric)
    dbscan = DBSCAN_Word2Vec(size, eps=eps, n_jobs=n_jobs, metric=metric)
    
    dbscan.fit_transform(vectors)
    print("DBSCAN model generated for the size of word2vec", size, "and the eps",eps)
    
    model=np.load("dbscan"+str(size)+"_"+str(eps)+".npy")
    
    dataComplete=dbscan.createNewData(dataComplete, vocab, model, word2Vec)
    
    dataComplete.to_json(path_or_buf="DataProcessed_google"+str(size)+"_"+str(eps)+".json",orient='records')
    print("Data for the size",size," of word2vec and the eps",eps,"for the DBSCAN Completed")

print("All the DBSCAN model generation Complete")

