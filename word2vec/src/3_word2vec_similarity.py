#!/usr/bin/env python3
# coding: utf-8
# File: 3_word2vec_similarity.py
# Author: lxw
# Date: 12/24/17 5:49 PM
"""
使用训练好的word2vec模型，计算两个句子的相似度

### References
[Word2Vec(三)-模型训练和计算余弦距离](http://blog.chatbot.io/development/2017/07/29/cosine-similarity-2/)
"""

import gensim
import numpy as np
import os
import time


class SimilarityCalculator:
    def __init__(self, model_file, binary=False):
        # 1. Loading model
        start_time = time.time()
        self.model = self._load_model(model_file=model_file, binary=binary)
        end_time = time.time()
        print("Model load time cost:{0}".format(end_time - start_time))    # 143.73s

        # 2. DIMENSION_SIZE used for OOV(Out Of Vocabulary).
        # self.DIMENSION_SIZE = os.environ["DIMENSION_SIZE"] if "DIMENSION_SIZE" in os.environ else 200

        # 3. lambdas for cos similarity
        self.roll_up_vecs = lambda x: np.sum(x, axis=0)    # axis=(0: column, 1: row)
        self.norm_of_vec = lambda x: np.sqrt(np.sum(np.square(x)))    # 向量的模

    def _load_model(self, model_file="../data/corpus.model.bin", binary=False):
        """
        Load model with C format word2vec file.
        """
        if not os.path.exists(model_file):
            raise Exception("Model file does not exist.")
        model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=binary, unicode_errors="ignore")
        return model

    def similarity_distance(self, s1, s2):
        return self._similarity_distance_model(s1, s2, model=self.model)

    def _similarity_distance_model(self, sentence1, sentence2, model):
        """
        compute cosine similarity of v1 to v2: (v1 dot v2)/(||v1||*||v2||)
        """
        def _vector(sentence):
            vectors = []
            for _index, word in enumerate(sentence.split()):    # NOTE: `sentence` must be in form of segmentations(space separated).
                try:
                    word = word.strip()
                    if word:
                        vec = model.wv[word]
                        vectors.append(vec)
                except KeyError as ke:
                    # vectors.append(np.zeros(self.DIMENSION_SIZE, dtype=float))
                    # `continue` is better?: we do not need to use DIMENSION_SIZE, and do not need to care the dimension size of the model.
                    continue

            return vectors

        vec1 = _vector(sentence1)    # <list of ndarray>: shape: 4 * (200,)
        vec2 = _vector(sentence2)    # <list of ndarray>: shape: 6 * (200,)
        vec1_rolled_up = self.roll_up_vecs(vec1)    # <ndarray> shape: (200,)
        vec2_rolled_up = self.roll_up_vecs(vec2)    # <ndarray> shape: (200,)
        # denominator1 = sim_denominator(vec1)    # NO: the numerator is np.dot(vec1_rolled_up, vec2_rolled_up)
        # denominator2 = sim_denominator(vec2)    # NO
        denominator1 = self.norm_of_vec(vec1_rolled_up)    # 74.1745
        denominator2 = self.norm_of_vec(vec2_rolled_up)    # 98.9726

        similarity = np.dot(vec1_rolled_up, vec2_rolled_up) / (denominator1 * denominator2)
        # np.dot(vec1_rolled_up, vec2_rolled_up): 4987.01
        similarity = "{0:.3f}".format(similarity)    # 0.679
        return similarity


def run():
    sc = SimilarityCalculator("../data/word_vector.bin")
    sentences = ["登录 不 上去 怎么办 ", "扫码 一直 不 能 成功 怎么办"]    # NOTE: `sentences` must be in form of segmentations.
    print(sc.similarity_distance(sentences[0], sentences[1]))


if __name__ == "__main__":
    run()
