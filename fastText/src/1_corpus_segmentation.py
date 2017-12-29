#!/usr/bin/env python3
# coding: utf-8
# File: 1_corpus_segmentation.py
# Author: lxw
# Date: 12/28/17 3:46 PM
"""
对673w + 200w语料数据进行分词，生成用于训练word vectors的corpus
"""

import configparser
import jieba
import pymongo
import pymysql
import time

from public_utils import generate_logger


class SegmentCorpus():
    def __init__(self):
        self._load_config()

        # logging
        self.out_log = generate_logger("segmentcorpus_out")
        self.err_log = generate_logger("segmentcorpus_err")

    def __del__(self):
        # https://stackoverflow.com/questions/18401015/how-to-close-a-mongodb-python-connection
        print("in __del__()")
        self.conn.close()

    # 1. fetch 673w news from mongo.
    def fetch_mongo_news(self):
        """
        从mongo中【读取新闻->分词->写入文件】的操作太耗时了，所以把这个过程拆分开:先读取新闻写入到文件，然后在服务器上跑分词，并写入文件
        NOTE: 生成的文件太大， 必须放到/mnt目录下，不能放到当前目录，所以在当前目录下建立了/mnt目录的链接
        """
        with open("../data/mnt_link/673w_news_id_title_content.txt", "wb") as f:
            for news in self.col.find(no_cursor_timeout=True):
                _id = str(news["_id"]).strip()
                title = news["news_title"].strip()
                content = news["news_content"].strip()
                f.write("{0}|lxw|{1}|lxw|{2}\n".format(_id, title, content).encode("utf-8"))

    # 2. 针对"../data/mnt_link/673w_news_id_title_content.txt"文件进行分词
    def segment_mongo_news(self):
        jieba.load_userdict("../data/steel_industry_dict.txt")
        f1 = open("../data/mnt_link/673w_news_id_title_content_seg.txt", "wb")
        with open("../data/mnt_link/673w_news_id_title_content.txt") as f:
            for line in f:
                line_list = line.split("|lxw|")
                if len(line_list) > 3:
                    self.err_log.error(line)
                    continue
                _id, title, content = line_list[0], line_list[1], line_list[2]
                title = jieba.cut(title)    # title: <generator object Tokenizer.cut at 0x7fccc3c88990>
                title = map(str.strip, title)
                title = [item for item in title if item != ""]
                title = " ".join(title) + "\n"

                content = jieba.cut(content)    # content: <generator object Tokenizer.cut at 0x7fccc3c88990>
                content = map(str.strip, content)
                content = [item for item in content if item != ""]
                content = " ".join(content) + "\n"
                title_content = title + content
                f1.write(title_content.encode("utf-8"))

    # 1'. fetch 200w news from mysql.
    def fetch_mysql_news(self):
        """
        从mysql中【读取新闻->分词->写入文件】的操作太耗时了，所以把这个过程拆分开:先读取新闻写入到文件，然后在服务器上跑分词，并写入文件
        
        Reference: [使用python遍历mysql中有千万行数据的大表](https://www.jianshu.com/p/80b81a68fd72)
        """
        with open("../data/200w_news_id_title_content.txt", "wb") as f:
            sql = "SELECT id, title, content FROM {0};".format(self.mysql_table)
            try:
                self.cursor.execute(sql)
                results = self.cursor.fetchall()
                for row in results:
                    _id = row[0]
                    title = row[1].strip()
                    content = row[2].strip()
                    f.write("{0}|lxw|{1}|lxw|{2}\n".format(_id, title, content).encode("utf-8"))
            except Exception as e:
                print("Error: unable to fetch data. {}".format(e))

    # 2'. 针对"../data/673w_news_title_content.txt"文件进行分词
    def segment_mysql_news(self):
        jieba.load_userdict("../data/steel_industry_dict.txt")
        f1 = open("../data/673w_news_title_content_seg.txt", "wb")
        with open("../data/673w_news_title_content.txt") as f:
            for line in f:
                line = jieba.cut(line.strip())  # line: <generator object Tokenizer.cut at 0x7fccc3c88990>
                line = map(str.strip, line)
                line = [item for item in line if item != ""]
                line = " ".join(line) + "\n"
                f1.write(line.encode("utf-8"))

    def _load_config(self):
        config = configparser.ConfigParser()
        config.read("../data/.fastTextConf.ini")

        # MongoDB
        mongodb = config["MongoDB"]
        self.conn = pymongo.MongoClient(mongodb["host"], int(mongodb["port"]))
        self.db = self.conn[mongodb["db"]]
        self.col = self.db[mongodb["collection"]]

        # MySQL
        mysql = config["MySQL"]
        mysql_conn = pymysql.connect(mysql["host"], mysql["user"], mysql["passwd"], mysql["db"])
        self.cursor = mysql_conn.cursor()
        self.mysql_table = mysql["table"]


if __name__ == "__main__":
    start_time = time.time()
    sc = SegmentCorpus()
    # 1. fetch 673w news from mongo.
    sc.fetch_mongo_news()

    # 2. 针对"../data/mnt_link/673w_news_id_title_content.txt"文件进行分词
    # sc.segment_mongo_news()

    # 1'. fetch 200w news from mysql.
    # sc.fetch_mysql_news()

    end_time = time.time()
    print("Time cost:{0:.3f}".format(end_time - start_time))
