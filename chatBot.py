#%%
# coding: utf-8
# from __future__ import unicode_literals, print_function, division

from io import open
import random
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from gensim import models
from gensim.models.wrappers import FastText
import jieba
#%%
""" Set device """
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
#%%
""" pre-train word embedding """
modelPath = R'C:\Users\kobe24\Desktop\PreTrain_Model\pretrain_fasttext_cc_zh_300.bin'
model = FastText.load_fasttext_format(modelPath)

#%%
""" pre-train word embedding ()"""
modelPath = R'Gossiping_d300_count5.model'
model = models.Word2Vec.load(modelPath)

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
for i in range(10):
    print(qa_pair[i])


#%%

# 隨機取 pair
training_pairs = [(random.choice(qa_pair)) for i in range(10)]
training_pairs
#%%
testPair_ = [(random.choice(qa_pair)) for i in range(50)]
testPair_

#%%
def getSegmentEmbeddingTensor(sentence_pair):
    """ 取得字詞 Embedding 以 tensor 回傳(QA pair) """
    tensor_pair = list()
    for i in range(len(sentence_pair)):
        seg_list = jieba.lcut(sentence_pair[i])
        
        # print(seg_list)
        tensor_Emb = list()
        for text in seg_list:       # tensor_Emb = [model[text] for text in seg_list]
            try:
                tensor_Emb.append(model[text])
            except KeyError as msg:
                if len(tensor_Emb)>0:
                    tensor_Emb.append(tensor_Emb[len(tensor_Emb)-1])
                else:
                    tensor_Emb.append([0 for i in range(300)])
            
        # tensor_Emb = torch.FloatTensor(tensor_Emb)
        # tensor_Emb = torch.LongTensor(tensor_Emb)
        tensor_pair.append(tensor_Emb)

    # return tensor_pair[0],tensor_pair[1]    ## 0: 問題(len字),1: 回答(len字)
    return torch.FloatTensor(tensor_pair[0]),torch.FloatTensor(tensor_pair[1])    ## 0: 問題(len字),1: 回答(len字)    

#%%
""" example """
# print(":%s %s" % (qa_pair[0][0],qa_pair[0][1]))
q__,a__ = getSegmentEmbeddingTensor(qa_pair[0]) 
print(":%s %s" % (len(q__),len(a__)))
q__,a__

#%%
"""test list to np , np to tensor"""
tmp = np.asarray(q__)
tmp = torch.from_numpy(tmp)
tmp
#%%
"""test list to tensor"""
tmp2 = torch.FloatTensor(q__)
tmp2[0]


#%%
"""Encoder"""
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size)

    def forward(self, input, hidden = None):
        output, hidden = self.rnn(input, hidden)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

#%%
"""Decoder"""
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # self.embedding = nn.Embedding(output_size, hidden_size)
        # self.gru = nn.GRU(hidden_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden = None):
        # output = F.relu(output)
        output, hidden = self.rnn(input, hidden)
        # output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



#%%
hidden_size = 300
input_size = 300
n_iters = 5000

learning_rate=0.01

encoder = EncoderRNN(input_size, hidden_size).to(device)
decoder = DecoderRNN(hidden_size, hidden_size).to(device)

encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss()
#%%
""" Set train pair """
training_pairs = list()
for i in range(n_iters):
    # print(qa_pair[i])
    q_,a_ = getSegmentEmbeddingTensor(qa_pair[i])
    training_pairs.append([q_,a_])

#%%

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=300):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(300, encoder.hidden_size, device=device)

    loss = 0

    encoder_hidden = None
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            torch.unsqueeze(input_tensor[ei].view(1,300), dim=0).cuda()
            , encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]


    decoder_input = torch.tensor([[0.0 for i in range(300)]], device=device)
    decoder_hidden = encoder_hidden

    # print(decoder_hidden)

    for di in range(target_length):       
        decoder_output, decoder_hidden = decoder(
            torch.unsqueeze(decoder_input, dim=0).cuda() ,
            decoder_hidden)

        loss += criterion(decoder_output, target_tensor[di].cuda())
        decoder_input = target_tensor[di].view(1,300)  # Teacher forcing


    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


#%%
""" testing """
loss = train(input_tensor,target_tensor,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion)
loss
#%%

import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

#%%
start = time.time()

print_loss_total = 0  # Reset every print_every
print_every = 100
loss = 0

encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

criterion = nn.MSELoss()

for iter in range(1, n_iters + 1):
    training_pair = training_pairs[iter - 1]
    input_tensor = training_pair[0]
    target_tensor = training_pair[1]

    loss = train(input_tensor, target_tensor, encoder, 
                 decoder, encoder_optimizer, decoder_optimizer, criterion)
    # loss = train(input_tensor, target_tensor, encoder,
    #              decoder, encoder_optimizer, decoder_optimizer, criterion)

    print_loss_total += loss

    if iter % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                     iter, iter / n_iters * 100, print_loss_avg))

#%%
""" Store model"""
torch.save(encoder, 'encoder_Gossiping.pkl')  # Save model
torch.save(decoder, 'decoder_Gossiping.pkl')  # Save model

#%%
""" evluate """


""" Input Question """
testPair = qa_pair[50] ## testing
q_,a_ = getSegmentEmbeddingTensor(testPair)
input_tensor = q_
# input_tensor[0].view(1,300).size()

encoder_hidden = None
for index in range(len(input_tensor)):
    encoder_output, encoder_hidden = encoder(
        torch.unsqueeze(input_tensor[index].view(1,300), dim=0).cuda()
        , encoder_hidden)

#%%
""" Decode Answer"""

decoder_input = torch.tensor([[0.0 for i in range(300)]], device=device)
decoder_hidden = encoder_hidden
pred_list = list()

for di in range(20):       
    decoder_output, decoder_hidden = decoder(
        torch.unsqueeze(decoder_input, dim=0).cuda() ,
        decoder_hidden)

    pred_list.append(decoder_output)


#%%
# show pred
mywv = pred_list[0].view(300).cpu().data.numpy()

#%%
testPair

#%%
predWord = list()
for di in range(len(pred_list)):
    mywv = pred_list[di].view(300).cpu().data.numpy()
    predWord.append([text for text,similarity in model.most_similar(positive=[mywv], topn=5)])
#%%
predWord


#%% 
"""Below : For testing """


"""test"""
encoder_hidden = encoder.initHidden()
encoder_output,encoder_hidden = encoder(torch.unsqueeze(training_pairs[0][0][0].view(1,300), dim=0).cuda())

#%%
len(training_pairs)
len(training_pairs[0])      ## Q or A
len(training_pairs[0][0])   ## 第幾個句子
len(training_pairs[0][0][0])   ## 該句第幾個字


#%% test
training_pair = training_pairs[0]   ## 取 第一組PAIR
input_tensor = training_pair[0] ## 取 Q (共11字)
input_tensor[0] ## Q sentence 的第一字

#%%
encoder_output,encoder_hidden = encoder(torch.unsqueeze(input_tensor[0].view(1,300), dim=0).cuda())


#%%
input_length = input_tensor.size(0)
input_length
#%%
encoder_outputs = torch.zeros(300, encoder.hidden_size, device=device)
encoder_outputs
#%%

encoder_hidden = None
for ei in range(input_length):
    encoder_output, encoder_hidden = encoder(
        torch.unsqueeze(input_tensor[ei].view(1,300), dim=0).cuda()
        , encoder_hidden)
    encoder_outputs[ei] = encoder_output[0, 0]


#%%
torch.unsqueeze(input_tensor[0].view(1,300), dim=0).size()
#%%
for index in range(len(encoder_outputs)):
    print(encoder_outputs[index])


#%%
training_pair = training_pairs[0]   ## 第一組 PAIR
target_tensor = training_pair[1]    ## 取 A (共15字)
len(target_tensor) 

#%%
SOS_token = 0
decoder_input = torch.tensor([[0.0 for i in range(300)]], device=device)
decoder_input.size()

#%%
decoder_hidden = encoder_hidden
decoder_hidden

#%%
decoder_output, decoder_hidden = decoder(
    torch.unsqueeze(decoder_input, dim=0).cuda()
    , decoder_hidden)
