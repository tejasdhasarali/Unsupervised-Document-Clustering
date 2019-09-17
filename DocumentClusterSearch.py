# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit
from scipy.stats import randint as sp_randint
from sklearn.pipeline import Pipeline
from random import sample
from datetime import datetime
from functools import lru_cache
from urllib.parse import urlparse


class DBSCAN_Fit(DBSCAN):
    
    def transform(self, X, y=None, *args, **kwargs):
        return super(DBSCAN, self).fit_predict(X, *args, **kwargs)


class DBSCAN_Score():
    
    def fit(self, X, *args, **kwargs):
        return self
    
    def print_groups(self, pred, duplicates=False, over_write=True):
        groups = {}
        for i, p in enumerate(pred):
            if p not in groups:
                groups[p] = []
            groups[p].append(i)
        if -1 in groups:
            del groups[-1]
        r_groups = {}
        for group, docs in groups.items():
            for doc in sorted(docs):
                if group not in r_groups:
                    r_groups[group] = [doc]
                else:
                    if r_groups[group][-1] + 1 != doc or duplicates:
                        r_groups[group].append(doc)
                    else:
                        r_groups[group][-1] = doc
        
        groups = {}
        group_size = []
        for group, docs in r_groups.items():
            if len(docs) <= 1:
                continue
            groups[group] = docs
            group_size.append((len(docs), group))
        for windowSize, groupSize, samples, fileType in (
                (140, 20, 10, 'top'), 
                (None, float('inf'), float('inf'), 'full')):
            name_append = 0
            if not over_write:
                name_append = int(datetime.now().timestamp())
            with open('{}_{}_{}.txt'.format('', fileType,
                      name_append), 'w', encoding='utf-8') as outFile:
                for i, (size, group) in enumerate(sorted(group_size)[::-1]):
                    if i > groupSize: break
                    outFile.write('Group {} - Size {}\n\n'.format(group, size))
                    docs = sample(groups[group], min(len(groups[group]), samples))
                    for doc in docs:
                        outFile.write('Title   : ' + 
                                      df.loc[doc].pageTitle[:windowSize] + '\n')
                        outFile.write('Relation: ' + 
                                      df.loc[doc].relation[:windowSize].replace('\n', ' '))
                        outFile.write('\n')
                    outFile.write('=' * 151 + '\n')

    def score_metric(self, X):
        global df
        unique = {}
        unique_all = set()
        unique_all_all = set()
        grouped = 1
        for i, j in enumerate(X):
            
            if j not in unique:
                unique[j] = [0, set()]
            url = urlparse(df.loc[i].url)
            url = url.hostname
            unique[j][0] += 1
            unique[j][1].add(url)
            unique_all_all.add(url)
            if j != -1:
                unique_all.add(url)
                grouped += 1
				
        if -1 in unique:
            del unique[-1]
        
        return ( sum( len(u[1])/u[0] for k, u  in unique.items()) / (len(unique) + 500) )
        
    def score(self, X, y=None):
        print('Score shape {}'.format(X.shape))
        if X.shape[0] <= 1:
            return 0
        score = self.score_metric(X)
        global best_score
        if score > best_score:
            best_score = score
            self.print_groups(X)
        return score
    
    scorer_ = score


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


path = r'DataProcessed'
all_files = glob.glob(path + '*.json')
li = []
ctr = 0
ctr_max = 4
for filename in all_files:
    ctr += 1
    print(filename)
    df_p = pd.read_json(filename)
    li.append(df_p)
    if ctr >= ctr_max:
        break
df = pd.concat(li, axis=0, ignore_index=True)

data = df.relation.dropna()
print('Data all {}'.format(data.shape))

data = data[:1_000_000]
print('Data done {}'.format(data.shape))

params = {"dbscan__eps": (0.1, 0.5, 0.75, 0.9, 0.95,),
          "dbscan__min_samples": (2,20,200), # 2000,
          "dbscan__n_jobs": (1,),
          "tfidf__max_features": sp_randint(10_000, 800_000), # 1_000_000,
          "tfidf__ngram_range": ((1,1),(1,2),(1,4),(2,2),(2,4),(4,4)), #(1,4),(2,2),(2,4),(4,4),
          "tfidf__strip_accents": ('unicode',),
          "tfidf__stop_words": ('english',),
          }
steps = [('tfidf', TfidfVectorizer()), ('dbscan', DBSCAN_Fit()), ('score', DBSCAN_Score()),]
pipeline = Pipeline(steps)

split = [0 for _ in range(data_pts-1)] + [-1]
cv = PredefinedSplit(split)

cv_jobs = 1
random_search = RandomizedSearchCV(pipeline,param_distributions=params,
                                  n_iter=20, cv=cv, n_jobs=cv_jobs,
                                  pre_dispatch=cv_jobs)

random_search.fit(data)

report(random_search.cv_results_)
