#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 19:19:28 2019

@author: tejashasarali
"""

import pandas as pd
import spacy
import string
import os.path
from keras import backend as K

nlp = spacy.load('en', disable=['ner', 'parser'])
nlp.add_pipe(nlp.create_pipe('sentencizer'))

printable = set(list(string.printable))

columnsToDrop=['TableContextTimeStampAfterTable','TableContextTimeStampBeforeTable','hasHeader','hasKeyColumn',
              'headerPosition','headerRowIndex','keyColumnIndex','lastModified','recordEndOffset','recordOffset',
              's3Link','tableNum','tableOrientation','tableType','title','textBeforeTable', 'textAfterTable']


def processRelation(columns):
    relation=columns[0]
    tableType=columns[1]
    string=""
    if(tableType=='ENTITY' or tableType=='RELATION' or tableType=='OTHER'):
        for rArray in zip(*relation):
            string+=" ".join(str(r) for r in rArray)+" "
    elif(tableType=='MATRIX' or tableType=='LAYOUT'):
        for rArray in relation:
            string+=" ".join(str(r) for r in rArray)+" "
    else:
        for rArray in relation:
            string+=" ".join(str(r) for r in rArray)+" "
    return string


def preProcess(columns):
    text=columns[0]
    temp=filter(lambda x: x in printable, text)
    text="".join(temp)
    text = nlp(text)
    text = ' '.join([ word.lemma_ if word.lemma_ != '-PRON-' else word.text
                     for word in text if not word.is_stop])
    return text


for i in range(31):
    if(os.path.isfile("Data"+str(i)+".json")):
        print("Data",i)
        data=pd.read_json("Data"+str(i)+".json")
    else:
        continue
    
    data['relation']=data[['relation','tableType']].apply(processRelation,axis=1)
    data=data.drop(columns=columnsToDrop)
    data['relation']=data[['relation']].apply(preProcess,axis=1)
    data.to_json(path_or_buf="DataProcessedsdfb "+str(i)+".json",orient='records')
    print("Completed ",i)
    K.clear_session()