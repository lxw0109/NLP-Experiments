#!/usr/bin/env python3
# coding: utf-8
# File: 2_use_fasttext_model.py
# Author: lxw
# Date: 1/2/18 6:07 PM

"""
### 注意：
训练模型可以使用fasttext命令行工具进行(../doc/fastText_train.png)，也可以使用本文件中的方法训练。
**但是**，使用fasttext命令行工具训练的模型，无法使用本文件使用的fasttext包导入。
 
训练和使用fasttext模型
Reference: https://pypi.python.org/pypi/fasttext

当使用gensim导入fastText训练的模型文件时，总是提示下面的错误：
`UnicodeDecodeError: 'utf8' codec can't decode byte 0x80 in position 32: invalid start byte`
This is due to the fact that the fastText binary file also contains information from subword units, which can be used to compute word vectors for out-of-vocabulary words.
解决方法参考：https://github.com/facebookresearch/fastText/issues/171. 但这种解决方法是使用`.vec`文件生成适配gensim的`.bin`文件，而且生成`.bin`文件的时间比较长，所以不推荐使用这种方法
此处针对这种方法的模型导入和使用方法略
"""

import fasttext    # Reference: https://pypi.python.org/pypi/fasttext    # 这个包无法导入使用
import time


# 1. Model training.
def train_fasttext_model():
    # Skipgram model
    # equals to: `./fasttext skipgram -input ../data/880w_news_title_content_seg_sort_uniq_head_2.txt -output lxw_model_sg`
    model_sg = fasttext.skipgram("../data/880w_news_title_content_seg_sort_uniq_head_2.txt", "../data/lxw_model_sg")
    # 自动生成文件../data/lxw_model_sg.bin 和 ../data/lxw_model_sg.vec
    print(model_sg.words)    # list of words in dictionary

    # CBOW model
    # equals to: `./fasttext cbow -input ../data/880w_news_title_content_seg_sort_uniq_head_2.txt -output lxw_model_sg`
    model_cbow = fasttext.cbow("../data/880w_news_title_content_seg_sort_uniq_head_2.txt", "../data/lxw_model_cbow")
    # 自动生成文件../data/lxw_model_cbow.bin 和 ../data/lxw_model_cbow.vec
    print(model_cbow.words)    # list of words in dictionary
    print(type(model_cbow.words))    # <class 'set'>
    # NOTE: 生成的两个.vec文件针对同一个词的向量是不同的

    print("intersection:{}".format(model_sg.words - model_cbow.words))    # intersection:set()
    print("intersection:{}".format(model_cbow.words - model_sg.words))    # intersection:set()


# 2. Using the pre-trained model.
def use_fasttext_model():
    # 训练模型可以使用fasttext命令行工具进行(../doc/fastText_train.png)，也可以使用本文件使用的fasttext包训练。但是，
    # 使用fasttext命令行工具训练的模型，无法使用本文件使用的fasttext包导入
    model = fasttext.load_model("../data/lxw_model_sg.bin", encoding="utf-8")
    # print(model.words)
    print(model["先生"])
    print(model["刘晓伟"])   # OOV
    print(model["陈贺"])   # OOV

    model = fasttext.load_model("../data/lxw_model_cbow.bin", encoding="utf-8")
    # print(model.words)
    print(model["先生"])
    print(model["刘晓伟"])   # OOV
    print(model["陈贺"])   # OOV
    # NOTE: 简单的测试发现, 两个不同的模型针对同一个OOV计算得到的向量是一样的(与pyfasttext包的情况相同，详情可参见3_use_fasttext_model), 非OOV的向量是不一样的。


def main():
    # 1. Model training.
    # train_fasttext_model()

    # 2. Using the pre-trained model.
    use_fasttext_model()


if __name__ == '__main__':
    main()