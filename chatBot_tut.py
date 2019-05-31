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
import math


from gensim import models
from gensim.models.wrappers import FastText
import gensim
import jieba
#%%
import math

#%%
""" Set device """
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

#%%
""" pre-train word embedding """


# modelPath = R'C:\Users\kobe24\Desktop\PreTrain_Model\pretrain_fasttext_cc_zh_300.bin'
# model = FastText.load_fasttext_format(modelPath)

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

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
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
qa_pair[0][0]


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
def getSegment(qa_pair):
    """ ````` """
    sentence_pair = list()
    for i in range(len(qa_pair)):
        seg_list = jieba.lcut(qa_pair[i])
        sentence = list()
        for text in seg_list:       # tensor_Emb = [model[text] for text in seg_list]
            sentence.append(text)

        sentence_pair.append(sentence)

    return sentence_pair

def getCharSegment(qa_pair):
    sentence_pair = list()
    for i in range(len(qa_pair)):
        sentence = list()
        for text in qa_pair[i]:
            sentence.append(text)
        sentence_pair.append(sentence)
    return sentence_pair
    

# t = getSegment(qa_pair[0])
# t
#%%
len(qa_pair)

#%%
myVoc = Voc('Gops')
sample_qa_pair = qa_pair[:20000]       ## 取樣
for single_pair in sample_qa_pair:
    # q_,a_ = getSegment(single_pair)
    q_,a_ = getCharSegment(single_pair)
    # print(q_,a_)
    myVoc.addSentence(q_)
    myVoc.addSentence(a_)


#%%
myVoc.word2index
#%%
myVoc.word2count
#%%
myVoc.word2index['語']

#%%
testSen = sample_qa_pair[0][1]
testSen
#%%
testSen.split('')
#%%
t = myVoc.word2index[word] for word in testSen.split(' ')

#%%
""" Prepare data for model """

def indexesFromSentence(voc, sentence):
    # return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]
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
def batch2TrainData(voc, pair_batch):
    # pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    pair_batch.sort(key=lambda question: len(question[0]), reverse=True) ## for chinese char
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len

#%%
sample_qa_pair
#%%
pair_batch = [random.choice(sample_qa_pair) for _ in range(3)]
pair_batch
#%%
# pair_batch.sort(key=lambda question: len(question[0].split(" ")), reverse=True)
pair_batch.sort(key=lambda question: len(question[0]), reverse=True)
pair_batch
#%%
input_batch, output_batch = [], []
for pair in pair_batch:
    input_batch.append(pair[0])
    output_batch.append(pair[1])
input_batch
#%%
inp, lengths = inputVar(input_batch, myVoc)
output, mask, max_target_len = outputVar(output_batch, myVoc)

#%%
inp
#%%
myVoc.word2index['什']




#%%
# Example for validation
small_batch_size = 64
batches = batch2TrainData(myVoc, [random.choice(sample_qa_pair) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

#%%
print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)


#%%
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
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden

#%%
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

#%%
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        # self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        # self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=1)

        # self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)

        """ attention version """
        # # Calculate attention weights from the current GRU output
        # attn_weights = self.attn(rnn_output, encoder_outputs)
        # # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        # context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # # Concatenate weighted context vector and GRU output using Luong eq. 5
        # rnn_output = rnn_output.squeeze(0)
        # context = context.squeeze(1)
        # concat_input = torch.cat((rnn_output, context), 1)
        # concat_output = torch.tanh(self.concat(concat_input))
        # # Predict next word using Luong eq. 6
        # output = self.out(concat_output)
        # output = F.softmax(output, dim=1)

        # output = torch.tanh(rnn_output)
        output = self.out(rnn_output)
        # output = F.softmax(rnn_output, dim=1).squeeze(0)

        # Return output and final hidden state
        return rnn_output, hidden

#%%
def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


#%%

""" Train testing """
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(myVoc.num_words, hidden_size)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN( embedding, hidden_size, myVoc.num_words, decoder_n_layers, dropout)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)

#%%
# Example for validation
small_batch_size = 64
batches = batch2TrainData(myVoc, [random.choice(sample_qa_pair) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches
#%%
learning_rate = 0.0001
decoder_learning_ratio = 5.0
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

#%%
""" training example"""

# Zero gradients
encoder_optimizer.zero_grad()
decoder_optimizer.zero_grad()

# Set device options
input_variable = input_variable.to(device)
lengths = lengths.to(device)
target_variable = target_variable.to(device)
mask = mask.to(device)

# Initialize variables
loss = 0
print_losses = []
n_totals = 0

#%%
# Forward pass through encoder
encoder_outputs, encoder_hidden = encoder(input_variable, lengths)
#%%
decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
decoder_input = decoder_input.to(device)
decoder_hidden = encoder_hidden[:decoder.n_layers]
#%%
decoder_input.size()
#%%
decoder_output, decoder_hidden = decoder(
    decoder_input, decoder_hidden, encoder_outputs
)
#%%
decoder_output.size()
#%%
decoder_output
#%%
mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[0], mask[0])
#%%
mask_loss, nTotal
#%%
mask_loss.item()
#%%
batches = batch2TrainData(myVoc, [random.choice(sample_qa_pair) for _ in range(6)])
input_variable, lengths, target_variable, mask, max_target_len = batches
lengths

#%%
encoder_outputs.size()

#%%


MAX_LENGTH=30

def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal


            # if(t==0):
            #     # print('=============')
            #     # print(decoder_input, decoder_hidden, encoder_outputs)
            #     # print(decoder_output.size())

            #     # """

            #     # decoder output is NaN!!

            #     # """

            #     print('=============')
            #     # print(decoder_output, target_variable[t], mask[t])
            #     print(mask_loss)
            #     print(loss)          

            #     if(math.isnan(mask_loss)):
            #         print('=============')
            #         print(decoder_output, target_variable[t], mask[t])
                
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    
    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


#%%

n_iteration = 4000
batch_size = 64
# Load batches for each iteration
training_batches = [batch2TrainData(myVoc, [random.choice(sample_qa_pair) for _ in range(batch_size)]) for _ in range(n_iteration)]

#%%
# Initializations
print('Initializing ...')
start_iteration = 1
print_loss = 0
print_every = 100
clip = 50.0
teacher_forcing_ratio = 1.0
#%%
# Training loop
print("Training...")
for iteration in range(start_iteration, n_iteration + 1):
    training_batch = training_batches[iteration - 1]
    # Extract fields from batch
    input_variable, lengths, target_variable, mask, max_target_len = training_batch

    # Run a training iteration with batch
    loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                 decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
    print_loss += loss


    # Print progress
    if iteration % print_every == 0:
        print_loss_avg = print_loss / print_every
        print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
        print_loss = 0


#%%
max_target_len