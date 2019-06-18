# -*- coding: utf-8 -*-

import logging

from gensim.models import word2vec

def main():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.LineSentence("data\Gossiping-QA-Dataset-w2v-train-Segment.txt")
    model = word2vec.Word2Vec(
        sentences, size=250, workers=8, min_count = 4, window=3)

    #保存模型，供日後使用
    model.save("word2vec\model\Gossiping_w2v_model.model")

    #模型讀取方式
    # model = word2vec.Word2Vec.load("your_model_name")

if __name__ == "__main__":
    main()
