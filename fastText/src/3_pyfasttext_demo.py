#!/usr/bin/env python3
# coding: utf-8
# File: 3_use_fasttext_model.py
# Author: lxw
# Date: 1/4/18 10:41 AM
"""
### 注意：
训练模型可以使用fasttext命令行工具进行(../doc/fastText_train.png)，也可以使用本文件中的方法训练。
**并且**，使用fasttext命令行工具训练的模型，也能够使用本文件使用的pyfasttext包导入。而且，使用pyfasttext包导入模型的速度
要比使用gensim包导入模型要快得多。

pyfasttext:
https://pypi.python.org/pypi/pyfasttext
https://github.com/vrasneur/pyfasttext
"""

import time

from pyfasttext import FastText


# 1. Model training.
def train_pyfasttext_model():
    # Skipgram model
    model_sg = FastText()
    # equals to: `./fasttext skipgram -input ../data/880w_news_title_content_seg_sort_uniq_head_2.txt -output lxw_model_sg_pyfasttext`
    model_sg.skipgram(input="../data/880w_news_title_content_seg_sort_uniq_head_2.txt", output="../data/lxw_model_sg_pyfasttext")
    # 自动生成文件../data/lxw_model_sg_pyfasttext.bin 和 ../data/lxw_model_sg_pyfasttext.vec
    print(model_sg.words)    # list of words in dictionary

    # CBOW model
    model_cbow = FastText()
    # equals to: `./fasttext cbow -input ../data/880w_news_title_content_seg_sort_uniq_head_2.txt -output lxw_model_cbow_pyfasttext`
    model_cbow.cbow(input="../data/880w_news_title_content_seg_sort_uniq_head_2.txt", output="../data/lxw_model_cbow_pyfasttext")
    # 自动生成文件../data/lxw_model_cbow_pyfasttext.bin 和 ../data/lxw_model_cbow_pyfasttext.vec
    print(model_cbow.words)    # list of words in dictionary
    print(type(model_cbow.words))    # <class 'list'>
    # NOTE: 生成的两个.vec文件针对同一个词的向量是不同的


# 2. Using the pre-trained model.
def use_pyfasttext_model():
    # OK
    # 训练模型可以使用fasttext命令行工具进行(../doc/fastText_train.png)，也可以使用本文件使用的pyfasttext包训练。

    """
    # OK: 1. pyfasttext包训练的模型的导入
    model = FastText("../data/lxw_model_sg_pyfasttext.bin")
    print(model["先生"])     # type(model["先生"]): <class 'array.array'>
    print(model.get_numpy_vector("先生"))    # type: <class 'numpy.ndarray'>
    print(model["刘晓伟"])   # OOV
    print(model.get_numpy_vector("刘晓伟"))
    print(model["陈贺"])   # OOV
    print(model.get_numpy_vector("陈贺"))

    model = FastText("../data/lxw_model_cbow_pyfasttext.bin")
    print(model["先生"])
    print(model.get_numpy_vector("先生"))    # type: <class 'numpy.ndarray'>
    print(model["刘晓伟"])   # OOV
    print(model.get_numpy_vector("刘晓伟"))
    print(model["陈贺"])   # OOV
    print(model.get_numpy_vector("陈贺"))
    # NOTE: 简单的测试发现, 两个不同的模型针对同一个OOV计算得到的向量是一样的(与fasttext包的情况相同，详情可参见NO_2_use_fasttext_model), 非OOV的向量是不一样的。
    """

    # OK: 2. fasttext命令行工具训练出来的模型的导入
    model = FastText("../data/880w_fasttext_skip_gram.bin")
    print(model["先生"])     # type(model["先生"]): <class 'array.array'>
    print(model.get_numpy_vector("先生"))
    # print(model["刘晓伟"])   # OK. OOV
    # print(model["陈贺"])   # OK. OOV

    # Sentence and text vectors.
    sentence_vec = model.get_numpy_sentence_vector("刘晓伟 是 个 好人")
    print(sentence_vec)


    """
    # OK: 3. fasttext包训练出来的模型的导入
    model = FastText("../data/lxw_model_sg.bin")
    print(model["先生"])
    print(model["刘晓伟"])   # OOV
    print(model["陈贺"])   # OOV

    model = FastText("../data/lxw_model_cbow.bin")
    print(model["先生"])
    print(model["刘晓伟"])   # OOV
    print(model["陈贺"])   # OOV
    # NOTE: 如果使用pyfasttext导入fasttext包训练的模型，计算出来的OOV都是0，没有什么意义
    """


def main():
    # 1. Model training.
    # train_pyfasttext_model()

    # 2. Using the pre-trained model.
    use_pyfasttext_model()


if __name__ == "__main__":
    main()
