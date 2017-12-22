# BasicsOnNLP
**自然语言处理(Natural Language Processing, NLP)** 基础理论的学习与代码实现（基于Sougou和Wiki-ZH数据集）  

## Requirements
本项目所有代码实现均基于[Python3.6+](https://www.python.org/downloads/)完成，所需要的Python包如`requirements.txt`文件，
请使用`pip install -r requirements.txt -i https://pypi.douban.com/simple/`命令进行安装(推荐使用[virtualenv + virtualenvwrapper](http://www.jianshu.com/p/44ab75fbaef2)管理Python虚拟环境)。

## 说明
1.每个目录包含某个独立知识点的相关代码实现，目录结构如下(以`word2vec`目录为例)  
```bash
word2vec
├── data
├── README.md
└── src
    ├── 1_cosine_similarity.py
    ├── 2_gensim_demo.py
    └── 3_word2vec_similarity.py
```
每个目录中均包含`data/`, `src/`, `README.md`三个文件，`data`为该知识点需要使用到的数据文件，`src`为相关的代码文件，
`README.md`为说明文件。  
对于`src/`目录而言，若该知识点涉及多个部分的代码实现，则将各文件按照`"序号_文件名"`的方式进行命名，以便于顺序查阅。 
