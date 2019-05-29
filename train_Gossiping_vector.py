#%%
# -*- coding: utf-8 -*-
import logging
from gensim.models import word2vec
import jieba
from gensim import models

#%%
def main():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.LineSentence("data\Gossiping-QA-Dataset-w2v-train-Segment.txt")
    model = word2vec.Word2Vec(
        sentences, size=300, workers=6, min_count=5, window=5)

    #保存模型，供日後使用
    model.save("Gossiping_d300_count5.model")

    #模型讀取方式
    # model = word2vec.Word2Vec.load("your_model_name")

#%%

def segment():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # jieba custom setting.
    jieba.set_dictionary('jieba_dict\dict.txt.big')

    # load stopwords set
    stopword_set = set()
    with open('jieba_dict\stopwords.txt','r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))
        pass        

    """
    以字元斷詞
    """
    wordSeg = False

    output = open('data\Gossiping-QA-Dataset-w2v-train-Segment.txt', 'w', encoding='utf-8')
    with open('data\Gossiping-QA-Dataset-w2v-train.txt', 'r', encoding='utf-8') as content :
        for texts_num, line in enumerate(content):
            line = line.strip('\n')
            
            if (wordSeg):
                charList = list(line)                       # 以字元斷詞
                for char in charList:
                    if (char != ' '):                       ## 非空格
                        output.write(char + ' ')                                    
                pass
            else:
                words = jieba.cut(line, cut_all=False)      # 結巴斷詞用
                for word in words:
                    if word not in stopword_set:
                        output.write(word + ' ')
                pass

            output.write('\n')

            if (texts_num + 1) % 10000 == 0:
                logging.info("已完成前 %d 行的斷詞" % (texts_num + 1))
    output.close()

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
qa_pair[0][0]


#%%
""" Create cropus """
def CreateCropus():
    filename = R'data/Gossiping-QA-Dataset-w2v-train.txt'
    output = open(filename, 'w', encoding='utf-8')

    for q_,a_ in qa_pair:
        output.write(q_+'\n'+a_+'\n')

    output.close()


#%%
if __name__ == "__main__":
    main()



#%%
modelPath = R'C:\Users\kobe24\Desktop\PreTrain_Model\pretrain_fasttext_cc_zh_300.bin'
model = models.Word2Vec.load('Gossiping_d300_count5.model')
#%%
if model['肥宅']

#%%
if '台灣' in model.wv.vocab:
    print('yes')
else:
    print('no')

#%%
model.most_similar('肥宅', topn=5)
