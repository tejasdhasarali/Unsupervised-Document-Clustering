
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from gensim.models import Word2Vec
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


class WordToVector:
    
    def __init__(self, min_count=5, size=100, window=5, workers=3, sg=0):
        self.min_count=min_count
        self.size=size
        self.window=window
        self.workers=workers
        self.sg=sg
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        model=Word2Vec(X, min_count=self.min_count, size=self.size, window=self.window,
                            workers=self.workers, sg=self.sg)
        return model
    
    def fit_transform(self, X, y=None):
        model=Word2Vec(X, min_count=self.min_count, size=self.size, window=self.window,
                            workers=self.workers, sg=self.sg)
        return model


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
            with open('{}_{}_{}_{}.txt'.format(fileType,str(self.size),str(self.eps),int(datetime.now().timestamp())),
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
            self.groupToCount[group].append(word2vec.wv.vocab[word].count)
            self.groupToIndex[group].append(index)
            index+=1
            #print(groupToCount)
 
        self.wordToMaxWord={}
        for group, word in zip(dbscan, vocab):
            if(group !=-1):
                self.wordToMaxWord[word]=self.indexToWord[self.groupToIndex[group][np.argmax(self.groupToCount[group])]]
        #print(self.wordToMaxWord)   
        
    def save(self,dbscan):
        np.save("dbscan"+str(self.size)+"_"+str(self.eps),np.array(dbscan))
        
        


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
            if token.isalnum() and token not in vocabSet and token in word2Vec.wv.vocab:
                vectors.append(word2Vec.wv[token])
                vocab.append(token)
                vocabSet.add(token)
    print("Data creation for DBSCAN completed the new length is",len(vectors))
    return (vectors, vocab)
    

print("Reading the Data complete data for Word2Vec")
    
dataComplete=pd.DataFrame(columns=columns)

   
#Reads the Complete Data

for i in range(31):
    if(os.path.isfile("DataProcessed "+str(i)+".json")):
        data=pd.read_json("DataProcessed "+str(i)+".json")
        if 'relationTokens' not in data.columns:
            data['relationTokens']=data[['relation']].apply(tokenize,axis=1)
        if 'relation' in data.columns:
            data=data.drop('relation', axis=1)
        dataComplete=dataComplete.append(data,ignore_index=True, sort=False)
        print("DataProcessed",i,"read")
        
print("Reading of the Data Completed")


min_count=10
window=5
workers=10
sg=0
size=300

print("Starting to train the word2Vec model")

wv=WordToVector(min_count=min_count, size=size,
                   window=window, workers=workers, sg=sg)
wvOutput=wv.fit_transform(dataComplete['relationTokens'].values)


wvOutput.save("word2vec "+str(size)+".model")

wvOutput=None

print("The word to vector model is saved")

dataComplete=dataComplete[:1200000]

vocab=set()
word2Vec = Word2Vec.load("word2vec "+str(size)+".model")
print(word2Vec)
    
if not (vocab):
    vectors, vocab = createDataForDBSCAN(word2Vec, dataComplete)
    
eps=0.25
n_jobs = 1
metric='cosine'

print("Starting to train the DBSCAN model with eps=",eps,"size=",size,"n_jobs=",n_jobs,"metric",metric)

dbscan = DBSCAN_Word2Vec(size, eps=eps, n_jobs=n_jobs, metric=metric)
dbscan.fit_transform(vectors)

print("DBSCAN model generated for the size of word2vec", size, "and the eps",eps)

model=np.load("dbscan"+str(size)+"_"+str(eps)+".npy")

dataComplete=dbscan.createNewData(dataComplete, vocab, model, word2Vec)
dataComplete.to_json(path_or_buf="DataProcessed "+str(size)+"_"+str(eps)+".json",orient='records')

print("Data for the size",size," of word2vec and the eps",eps,"for the DBSCAN Completed")

