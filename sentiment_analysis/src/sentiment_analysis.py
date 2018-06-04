#!/usr/bin/env python3
# coding: utf-8
# File: sentiment_analysis.py
# Author: lxw
# Date: 5/14/18 9:52 PM
"""
References:
[如何用Python做情感分析?](https://www.jianshu.com/p/d50a14541d01)
"""

def en_sentiment_analysis():
    from textblob import TextBlob

    text = "I am happy today. I feel sad today."
    blob = TextBlob(text)
    print(blob.sentences)
    print(blob.sentences[0].sentiment)
    """
    Sentiment(polarity=0.8, subjectivity=1.0)
    # 情感极性0.8, 主观性1.0
    # The polarity score is a float within the range [-1.0, 1.0]. -1代表完全负面, 1代表完全正面
    # The subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective.
    """
    print(blob.sentences[1].sentiment)
    print(blob.sentiment)


def zh_sentiment_analysis():
    from snownlp import SnowNLP

    text = "我今天很快乐。我今天很愤怒。"
    snow = SnowNLP(text)
    print(snow.sentiments)    # 0.723761992420350

    for sentence in snow.sentences:
        print(sentence)
        print(SnowNLP(sentence).sentiments)
        """
        我今天很快乐
        0.971889316039116    # 正面情感的概率
        我今天很愤怒
        0.07763913772213482    # 正面情感的概率
        """


if __name__ == "__main__":
    # For English
    # en_sentiment_analysis()

    # For Chinese
    zh_sentiment_analysis()
