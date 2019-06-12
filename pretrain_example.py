#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools

from gensim import models
from gensim.models.wrappers import FastText
import gensim
import jieba
import math
import numpy as np

#%%
#%%
mylist = ['點心','電腦','貓咪','狗狗','電視']
matrix_len = myVoc.num_words
weights_matrix = np.zeros((matrix_len, 300))    # 初始化
words_found = 0
weights_matrix
#%%

for i, word in enumerate(mylist):
    try: 
        weights_matrix[i] = model[word]
        words_found += 1
    except KeyError as msg:
        # weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))
        print(msg)

#%%
weights_matrix
#%%
weights_matrix.shape

#%%
num_embeddings, embedding_dim = weights_matrix.shape
num_embeddings, embedding_dim


#%%
weight = torch.FloatTensor(weights_matrix)
embedding = nn.Embedding.from_pretrained(weight)
#%%
# Get embeddings for index 1
input = torch.LongTensor([1])
embedding(input)

#%%

input = torch.LongTensor([1])
embedding(input)


#%%
def prepareData():
    filename = R'data/Gossiping-QA-Dataset.txt'
    qa_pair = list()
    with open(filename , encoding ='utf-8') as f:
        content = f.readlines()
    
    for pair in content:
        tmp = pair.replace(' ','').replace('\n','').split('\t')
        qa_pair.append([tmp[0],tmp[1]])
    return qa_pair
    
qa_pair = prepareData()

#%%
for ids in range(500):
    print('Q:[%s]\tA:[%s]' % (re.sub('\W', '', qa_pair[ids][0]),re.sub('\W', '', qa_pair[ids][1])))


#%%
qa_pair[3][1]

#%%
print(re.sub('\W', '', qa_pair[3][1]))

#%%
