#!/usr/bin/env python3
# coding: utf-8
# File: 1_corpus_segmentation.py
# Author: lxw
# Date: 12/28/17 3:46 PM
"""
对673w + 200w语料数据进行分词，生成用于训练word vectors的corpus
"""

import jieba
import pymongo
import time


class SegmentCorpus():
    def __init__(self):
        self.conn = pymongo.MongoClient("192.168.1.36", 27017)
        self.db = self.conn["eventDriver"]
        self.col = self.db["news673"]

    def __del__(self):
        # https://stackoverflow.com/questions/18401015/how-to-close-a-mongodb-python-connection
        print("in __del__()")
        self.conn.close()

    def fetch_mongo_news(self):
        """
        从mongo中【读取新闻->分词->写入文件】的操作太耗时了，所以把这个过程拆分开:先读取新闻写入到文件，然后在服务器上跑分词，并写入文件
        """
        with open("../data/673w_news_title_content.txt", "wb") as f:
            for news in self.col.find(no_cursor_timeout=True):
                title = news["news_title"].strip()
                content = news["news_content"].strip()
                title_content = "{0}\n{1}\n".format(title, content).encode("utf-8")
                f.write(title_content)

    def segment_news(self):
        jieba.load_userdict("../data/steel_industry_dict.txt")
        f1 = open("../data/673w_news_title_content_seg.txt", "wb")
        with open("../data/673w_news_title_content.txt") as f:
            for line in f:
                line = jieba.cut(line.strip())    # line: <generator object Tokenizer.cut at 0x7fccc3c88990>
                line = map(str.strip, line)
                line = [item for item in line if item != ""]
                line = " ".join(line) + "\n"
                f1.write(line.encode("utf-8"))


if __name__ == "__main__":
    start_time = time.time()
    sc = SegmentCorpus()
    # 1. fetch 673w news from mongo.
    # sc.fetch_mongo_news()

    # 2. 针对"../data/673w_news_title_content.txt"文件进行分词
    sc.segment_news()
    end_time = time.time()
    print("Time cost:{0:.3f}".format(end_time - start_time))
