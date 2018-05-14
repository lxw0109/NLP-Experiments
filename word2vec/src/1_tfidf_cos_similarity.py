#!/usr/bin/env python3
# coding: utf-8
# File: 1_tfidf_cos_similarity.py
# Author: lxw
# Date: 12/22/17 9:54 AM
"""
使用TF-IDF算法提取文章(句子)的关键词，并计算两篇文章(句子)的余弦相似度

### References
[Word2Vec(一)-余弦相似性数学原理](http://blog.chatbot.io/development/2017/06/14/cosine-similarity/)
**注意**: 原文中有多处错误, 不推荐阅读原文T_T

主要思想：
1) 使用TF-IDF算法，找出两篇文章的**关键词**
2) 每篇文章各取出若干个关键词(如20个)，合并成一个集合，并得到每篇文章对于这个集合中的词的tf-idf值向量
3) 计算两个向量的余弦相似度，值越大就表示越相似
"""

import math
import jieba.analyse


def get_keywords(article):
    """
    :param article: <str>. 无需分词
    :return: <list of tuple>.
    """
    # 通过TF-IDF算法提取关键词
    res = jieba.analyse.extract_tags(sentence=article, topK=20, withWeight=True)    # 得到的tf-idf值是已经归一化了的，不需要再人工归一化(因为tf是做了归一化的)
    # print(res)    # <list of tuple>
    return res


def get_tfidf_vectors(res1=None, res2=None):
    """
    :param res1: <list of tuple>.
    :param res2: <list of tuple>.
    :return: vector1, vector2
    """
    # 向量，可以使用list表示
    vector1 = []
    vector2 = []
    # tf-idf 可以使用dict表示
    tf_idf1 = {i[0]: i[1] for i in res1}
    tf_idf2 = {i[0]: i[1] for i in res2}

    res = set(list(tf_idf1.keys()) + list(tf_idf2.keys()))    # set of terms.

    # 填充词频向量
    for word in res:
        if word in tf_idf1:
            vector1.append(tf_idf1[word])
        else:
            vector1.append(0)
        if word in tf_idf2:
            vector2.append(tf_idf2[word])
        else:
            vector2.append(0)

    print("vector1:{}".format(vector1))    # vector1: [2.85129420151, 1.039149057915, 0.756830171665, 0, 0]
    print("vector2:{}".format(vector2))    # vector2: [2.85129420151, 0, 0, 2.0508960413675, 2.4025545527075]
    return vector1, vector2


def numerator(vector1, vector2):    #分子
    return sum(Ai * Bi for Ai, Bi in zip(vector1, vector2))


def denominator(vector):    #分母
    # return math.sqrt(sum(a * b for a, b in zip(vector, vector)))    # Obviously bad.
    return math.sqrt(sum(Ai * Ai for Ai in vector))


def cosine_sim(vector1, vector2):
    # vector1 = [1.039149057915, 2.85129420151, 0.756830171665, 0, 0]
    # vector2 = [0, 2.85129420151, 0, 2.0508960413675, 2.4025545527075]
    return numerator(vector1, vector2) / (denominator(vector1) * denominator(vector2))


def main():
    article1 = "我喜欢中国，也喜欢美国。"
    # article2 = "我喜欢足球，不喜欢篮球。我喜欢足球，不喜欢篮球。我喜欢足球，不喜欢篮球。我喜欢足球，不喜欢篮球。"    # 0.610829503411293
    # article2 = "我喜欢足球，不喜欢喜欢喜欢篮球。"    # 0.7974529550358094
    article2 = "我喜欢足球，不喜欢篮球。"    # 0.610829503411293
    # article2 = "我好想你啊。"    # 0.0

    # 1) 使用TF-IDF算法，找出两篇文章的**关键词**
    res1 = get_keywords(article=article1)
    res2 = get_keywords(article=article2)

    # 2) 每篇文章的关键词合并成一个集合，并得到每篇文章对于这个集合中的词的tf-idf值向量
    vector1, vector2 = get_tfidf_vectors(res1=res1, res2=res2)

    # 3) 计算两个向量的余弦相似度，值越大就表示越相似
    print(cosine_sim(vector1=vector1, vector2=vector2))
    """
    math.cos(x): Return the cosine of x radians.
    math.acos(x): Return the arc cosine of x, in radians. [arc cosine: 反余弦]
    """


if __name__ == '__main__':
    main()