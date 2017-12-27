#!/usr/bin/env python3
# coding: utf-8
# File: 2_gensim_word2vec_train.py
# Author: lxw
# Date: 12/24/17 5:49 PM
"""
使用gensim包进行word2vec模型的训练和使用

### References
[基于python的gensim word2vec训练词向量](http://m.blog.csdn.net/lk7688535/article/details/52798735)
[gensim_word2vec.py](https://github.com/Roshanson/TextInfoExp/blob/master/Part4_Word_Similarity/word2vec/gensim_wors2vec.py)  
[Word2Vec（二）- gensim模块](http://blog.chatbot.io/development/2017/06/17/word2vec/)    不推荐阅读原文T_T
"""

import gensim
import jieba

from gensim.models import word2vec


def cut_words():
    """
    # 分词, 并将分词结果写入到文件
    """
    f1 = open("../data/corpus_seg.txt", "wb")
    with open("../data/corpus.txt") as f:
        for line in f:
            data = jieba.cut(line.strip())    # data: <generator object Tokenizer.cut at 0x7fccc3c88990>
            data = " ".join(data) + "\n"
            data = data.encode("utf-8")
            f1.write(data)
    f1.close()


def train_w2v():
    """
    训练word2vec模型, 并保存
    :return: model
    """
    sentences = word2vec.Text8Corpus("../data/corpus_seg.txt")    # 加载语料
    model = word2vec.Word2Vec(sentences, size=200, min_count=1, window=10)    # 训练skip-gram模型
    # NOTE: word2vec的参数意义和选择: https://github.com/lxw0109/NLPExperiments/blob/master/word2vec/doc/Learning%20Notes%20on%20word2vec.ipynb
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
    # NOTE: 句子的相似度能直接计算吗？ NO
    # sim_res = model.similarity("登录 不 上去 怎么办", "扫码 一直 不 能 成功 怎么办")    # KeyError: "word '登录 不 上去 怎么办' not in vocabulary"
    # sim_res = model.similarity("1 . 进行 资讯 重要性 算法 的 优化 讨论 ， 确定 下 一步 的 优化 路线", "通过 fastText 对 673w   +   200w 数据 重跑 一次 word2vec 的 模型")    # KeyError: "word '1 . 进行 资讯 重要性 算法 的 优化 讨论 ， 确定 下 一步 的 优化 路线' not in vocabulary"
    print(sim_res)

    for i in model.most_similar("进行"):
        print("{0}, {1}".format(i[0], i[1]))

    print(model[u"处理"])

    # NOTE: Compute OOV(Out Of Vocabulary) words
    # print(model["刘晓伟"])    # KeyError: "word '刘晓伟' not in vocabulary


if __name__ == '__main__':
    # 1. corpus分词
    cut_words()

    # 2. 使用corpus训练word2vec
    model = train_w2v()
    using_model(model)
