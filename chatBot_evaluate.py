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
import time

#%%
""" Set device """
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

#%%
""" Loading pre-train word embedding """

# modelPath = R'C:\Users\kobe24\Desktop\PreTrain_Model\word2vec.model'
# model = models.Word2Vec.load(modelPath)
modelPath = R'word2vec\model\Gossiping_w2v_model.model'
model = models.Word2Vec.load(modelPath)


#%%
def prepareData(filename = R'data\Gossiping-QA-Dataset.txt'):
    qa_pair = list()
    with open(filename , encoding ='utf-8') as f:
        content = f.readlines()
    
    for pair in content:
        tmp = pair.replace(' ','').replace('\n','').split('\t') ## split  Q && A
        if(len(tmp)==2):
            qa_pair.append([re.sub('\W', '', tmp[0]),re.sub('\W', '', tmp[1])]) ## Sentence 正規化

    return qa_pair
    
qa_pair = prepareData()

#%%

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {'PAD':0,'SOS':1,'EOS':2}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        # for word in sentence.split(' '):
        #     self.addWord(word)
        for word in sentence:
            self.addWord(word)        

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            if(word !='PAD' and word !='EOS' and word !='SOS'):
                self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)

#%%
def getSegment(qa_pair, mode='word'):
    """ mode can be word-level or char-level """
    sentence_pair = list()
    if(mode == 'word'):
        # word-level segent by Jieba
        for index in range(len(qa_pair)):
            seg_list = jieba.lcut(qa_pair[index])
            sentence = list()
            for text in seg_list:       # tensor_Emb = [model[text] for text in seg_list]
                sentence.append(text)
            sentence_pair.append(sentence)
    elif(mode == 'char'):
        # char-level segent by Jieba
        for index in range(len(qa_pair)):
            sentence = list()
            for text in qa_pair[index]:
                sentence.append(text)
            sentence_pair.append(sentence)
    
    return sentence_pair

#%%
myVoc = Voc('Gops')
sample_qa_pair = qa_pair       ## 取樣
for single_pair in sample_qa_pair:
    q_,a_ = getSegment(single_pair)
    myVoc.addSentence(q_)
    myVoc.addSentence(a_)

print(q_,a_)

#%%
# myVoc.index2word, myVoc.word2index, myVoc.word2count, myVoc.num_words

#%%
matrix_len = myVoc.num_words
weights_matrix = np.zeros((matrix_len, 300))    # 初始化
words_found = 0
weights_matrix

#%%
for index,word in myVoc.index2word.items():
    if(word == 'EOS' or word =='SOS'):
        weights_matrix[index] = np.random.uniform(low=-1, high=1, size=(300))   ## random
    elif(word == 'PAD'):
        weights_matrix[index] = np.zeros(300)   
    else:
        try: 
            weights_matrix[index] = model[word]
            words_found += 1
            if(words_found % 1000 == 0):
                print('Num %s , success.'% words_found)
        except KeyError as msg:
            # weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))
            # weights_matrix[index] = (weights_matrix[index-1] + model[myVoc.index2word[index + 1]])/2
            weights_matrix[index] = np.random.uniform(low=-1, high=1, size=(300))   ## random
            # print(msg)


#%%
num_embeddings, embedding_dim = weights_matrix.shape
num_embeddings, embedding_dim

#%%
weight = torch.FloatTensor(weights_matrix)
embedding = nn.Embedding.from_pretrained(weight).to(device)

#%%
""" 釋放 pretrain 資源 """
# model = None    

#%%
""" Prepare data for model (未改)"""   

def indexesFromSentence(voc, sentence ,mode = 'word' , MAX_LENGTH = 20):
    """ 可選 word level 或 char-level; 可設定 Max length """
    # return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]
    seg_list = jieba.lcut(sentence)
    seg_list = seg_list[:20]
    if(mode == 'word'):
        return [voc.word2index[word] for word in seg_list] + [EOS_token]
    elif (mode == 'char'):
        return [voc.word2index[word] for word in sentence] + [EOS_token]
    

def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch , MAX_LENGTH = 20):
    # pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    pair_batch.sort(key=lambda question: len(question[0]), reverse=True) ## for chinese char
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len

""" (以上未改) """
#%%
""" Seq2Seq Model"""
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        # pre-train fasttext embedding
        embedded = embedding(torch.cuda.LongTensor(input_seq))
        packed = embedded

        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden

# Luong attention layer
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word 'index'
        embedded = embedding(torch.cuda.LongTensor(input_step))
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


#%%
""" Evaluate Model """

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores

def evaluate(encoder, decoder, searcher, voc, sentence, max_length=20):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            # input_sentence = normalizeString(input_sentence)
            
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Q:', input_sentence)
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")
#%%
def SingleEvaluateInput(encoder, decoder, searcher, voc ,input_sentence):
    try:
        # Evaluate sentence
        output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
        # Format and print response sentence
        output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
        # print('Q:', input_sentence)
        print('Bot:', ' '.join(output_words))
        # BotAns = ('Bot:', ' '.join(output_words))
        # return BotAns
    except KeyError:
        print("Error: Encountered unknown word.")
        # return ("Error: Encountered unknown word.")

#%%
# evaluateInput(encoder, decoder, searcher, myVoc)

#%%
embedding = torch.load('chatbot_model\embedding_w2v_front30000')
encoder = torch.load('chatbot_model\encoder_pretrain_w2v_front30000')
decoder = torch.load('chatbot_model\decoder_pretrain_w2v_front30000')

encoder.eval()
decoder.eval()

#%%
searcher = GreedySearchDecoder(encoder, decoder)

#%%
for pair_ in qa_pair[:5]:
    print('=========================')
    print(pair_)
    input_sentence = pair_[0]
    SingleEvaluateInput(encoder, decoder, searcher, myVoc , input_sentence)

#%%
""" Testing """
test_qa_pair = prepareData(R'data\Gossiping-QA-Dataset.txt')
# sample_test_qa_pair = [random.choice(test_qa_pair[30000:]) for _ in range(100)]
sample_test_qa_pair = test_qa_pair[1000:5000]
# sample_test_qa_pair
#%%

with open(R'Result\pretrain_w2v_front30000_res.txt' ,'w', encoding ='utf-8') as output:
    for pair_ in sample_test_qa_pair:
        
        output.write('=========================\n')
        output.write(str(pair_))
        output.write('\n')
            
        # print('=========================')
        # print(pair_)

        input_sentence = pair_[0]
        botRes = SingleEvaluateInput(encoder, decoder, searcher, myVoc , input_sentence)
            
        output.write(str(botRes)+'\n')
    
output.close()


#%%
for pair_ in sample_test_qa_pair:  
    print('=========================')
    print(pair_)
#%%
for pair_ in sample_test_qa_pair:        
           
    print('=========================')
    print(pair_)

    input_sentence = pair_[0]
    SingleEvaluateInput(encoder, decoder, searcher, myVoc , input_sentence)
            

#%%
sample_test_qa_pair

#%%
input_sentence = 'IKEA為何總是這麼多人'
SingleEvaluateInput(encoder, decoder, searcher, myVoc , input_sentence)

#%%
print(encoder)
print(decoder)

#%%
台灣人是不是好操弄啊
學長問我覺得他是什麼樣的人
找不到工作怎麼辦

為什麼美國不要直接射飛彈暗殺金正恩
#%%
for index in range(10,100,10):
    filePath = '10times_model/Result/pretrain_w2v_front30000_res_{}.txt'.format(index)
    print(filePath)
    pass

#%%
