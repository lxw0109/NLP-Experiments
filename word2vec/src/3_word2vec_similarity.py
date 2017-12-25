#!/usr/bin/env python3
# coding: utf-8
# File: 3_word2vec_similarity.py
# Author: lxw
# Date: 12/24/17 5:49 PM
"""
### References
[Word2Vec（三） - 模型训练和计算余弦距离](http://blog.chatbot.io/development/2017/07/29/cosine-similarity-2/)

主要思想：
"""

import numpy as np
import os
import gensim

curdir = os.path.dirname(os.path.abspath(__file__))
print(curdir, os.path.curdir)

# for text format, can resolve vector size with the model file.
DIMENSION_SIZE = os.environ["DIMENSION_SIZE"] if "DIMENSION_SIZE" in os.environ else 100


def load_model(model_file = "../corpus.model.bin", binary=False):
    """
    Load model with C format word2vec file.
    """
    if not os.path.exists(model_file):
        raise Exception("Model file does not exist.")
    return gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=binary, unicode_errors="ignore")


# lambdas for cos similarity
sim_numerator = lambda x: np.sum(x, axis=0)    # 分子
sim_denominator = lambda x: np.sqrt(np.sum(np.square(x)))    # 分母


def similarity_distance(sentence1, sentence2, model):
    """
    compute cosine similarity of v1 to v2: (v1 dot v2)/(||v1||*||v2||)
    """
    def _vector(sentence):
        vectors = []
        for _index, word in enumerate(sentence.split()):
            try:
                word = word.decode("utf-8", errors="ignore").strip()
                if word:    # discard word if empty
                    vec = model.wv[word]
                    vectors.append(vec)
            except KeyError as ke:
                vectors.append(np.zeros(DIMENSION_SIZE, dtype=float))

        return vectors

    # todo, compute OOV words
    # print("v1", sentence1_vectors)
    # print("v2", sentence2_vectors)

    vec1 = _vector(sentence1)
    vec2 = _vector(sentence2)
    numerator1 = sim_numerator(vec1)
    numerator2 = sim_numerator(vec2)
    denominator1 = sim_denominator(vec1)
    denominator2 = sim_denominator(vec2)

    similarity = np.dot(vec1, vec2) / (denominator1 * denominator2)
    print(float("%.3f" % similarity))
    similarity1 = numerator1 * numerator2 / (denominator1 * denominator2)
    print(float("%.3f" % similarity))
    return float("%.3f" % similarity)


def run():
    txts = ["登录 不 上去 怎么办 ", "扫码 一直 不 能 成功 怎么办"]
    V = load_model(model_file="./w2v.bin", binary=True)
    print("loaded.")
    print(similarity_distance(txts[0], txts[1], V))


class SimilarityCalculator():
    """
    Similarity Calculator
    """
    def __init__(self, model_file, binary=False):
        self.model = load_model(model_file=model_file, binary=binary)

    def distance(self, s1, s2):
        return similarity_distance(s1, s2, model=self.model)


if __name__ == "__main__":
    run()