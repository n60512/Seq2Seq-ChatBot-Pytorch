import random

def prepareData():
    filename = R'data/Gossiping-QA-Dataset.txt'
    qa_pair = list()
    with open(filename , encoding ='utf-8') as f:
        content = f.readlines()
    
    wfilename = R'data/Gossiping-QA-Dataset_10000.txt'
    index = 0
    with open(wfilename ,'a', encoding = 'utf-8') as wfs:
        for pair in content:
            wfs.write(pair)
            index+=1
            if(index>10000):
                break        


    
prepareData()