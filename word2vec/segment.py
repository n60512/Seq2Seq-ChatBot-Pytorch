# -*- coding: utf-8 -*-

import jieba
import logging
import re

def prepareData(filename = R'data\Gossiping-QA-Dataset.txt'):
    qa_pair = list()
    with open(filename , encoding ='utf-8') as f:
        content = f.readlines()
    
    for pair in content:
        tmp = pair.replace(' ','').replace('\n','').split('\t') ## split  Q && A
        if(len(tmp)==2):    
            qa_pair.append([re.sub('\W', '', tmp[0]),re.sub('\W', '', tmp[1])]) ## Sentence 正規化

    return qa_pair
    



def main():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # jieba custom setting.

    qa_pair = prepareData()
    texts_num = 0
    text_row = [text for pair in qa_pair for text in pair]
    with open(R'data\Gossiping-QA-Dataset-segment.txt', 'w', encoding='utf-8') as output :
        for text in text_row:
            seg_list = jieba.lcut(text)
            for word in seg_list:
                output.write(word + ' ')
            output.write('\n')

            if (texts_num + 1) % 10000 == 0:
                logging.info("已完成前 %d 行的斷詞" % (texts_num + 1))        
            texts_num+=1        
    output.close()

if __name__ == '__main__':
    main()
