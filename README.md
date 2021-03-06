# NLP Experiments
**自然语言处理(Natural Language Processing, NLP)** 知识总结与代码实现(基于从新浪财经、东方财富、和讯网、人民网、中证网等网站爬取的新闻数据/Wiki-ZH数据/Sougou数据等数据集)。  

## Requirements
本项目所有代码实现均基于[Python3.6+](https://www.python.org/downloads/)完成，所需要的Python包如`requirements.txt`文件，
请使用`pip install -r requirements.txt -i https://pypi.douban.com/simple/`命令进行安装(推荐使用[Anaconda](https://docs.anaconda.com/anaconda/)/[virtualenv + virtualenvwrapper](http://www.jianshu.com/p/44ab75fbaef2)/管理Python虚拟环境)。

## 说明
1.每个目录包含某个独立知识点的相关代码实现，目录结构如下(以`word2vec`目录为例)  
```bash
word2vec
├── data
├── doc
└── src
    ├── 1_tfidf_cos_similarity.py
    ├── 2_gensim_word2vec_train.py
    └── 3_word2vec_similarity.py
```
每个目录中均包含`data/`, `src/`, `doc/`三个目录:  
+ `data/`为该知识点使用的数据&模型文件  
+ `doc/`为该知识点相关的总结文档  
+ `src/`为该知识点相关的代码文件  
 对于`src/`目录，若该知识点涉及多个部分的代码实现，则将各文件按照`"序号_文件名"`的方式进行命名，以便于顺序查阅。  
 
