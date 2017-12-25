#!/usr/bin/env python3
# coding: utf-8
# File: 2_gensim_similarity.py
# Author: lxw
# Date: 12/24/17 5:49 PM
"""
使用gensim包进行word2vec模型的训练和使用

### References
[基于python的gensim word2vec训练词向量](http://m.blog.csdn.net/lk7688535/article/details/52798735)
[gensim_word2vec.py](https://github.com/Roshanson/TextInfoExp/blob/master/Part4_Word_Similarity/word2vec/gensim_wors2vec.py)  
[Word2Vec（二）- gensim模块](http://blog.chatbot.io/development/2017/06/17/word2vec/)
"""

import gensim
import jieba

from gensim.models import word2vec


def cut_words():
    """
    # 分词, 并将分词结果写入到文件
    """
    with open("../data/corpus.txt") as f:
        content = f.read()
        data = jieba.cut(content)    # data: <generator object Tokenizer.cut at 0x7fccc3c88990>
        data = " ".join(data)

    with open("../data/corpus_seg.txt", "wb") as f:
        f.write(data.encode("utf-8"))


def train_w2v():
    """
    训练word2vec模型, 并保存
    :return: model
    """
    sentences = word2vec.Text8Corpus("../data/corpus_seg.txt")    # 加载语料
    model = word2vec.Word2Vec(sentences, size=200, min_count=1, window=5)    # 训练skip-gram模型
    # TODO: 1. word2vec的参数意义 2. word2vec的原理
    return model

    """
    # OK
    model.save("../data/corpus.model")    # 保存模型，以便重用
    model_load = word2vec.Word2Vec.load("../data/corpus.model")    # 加载模型
    using_model(model_load)

    # OK
    model.wv.save_word2vec_format("../data/corpus.model.bin", binary=False)
    model_load = gensim.models.KeyedVectors.load_word2vec_format("../data/corpus.model.bin")
    using_model(model_load)
    """


def using_model(model):
    # sim_res = model.similarity("好", "还行")    # KeyError: "word '好' not in vocabulary
    sim_res = model.similarity("进行", "处理")
    print(sim_res)

    for i in model.most_similar("进行"):
        print("{0}, {1}".format(i[0], i[1]))

    print(model[u"处理"])


if __name__ == '__main__':
    # 1. corpus分词
    cut_words()

    # 2. 使用corpus训练word2vec
    model = train_w2v()
    using_model(model)
