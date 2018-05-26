#encoding=utf-8
import pandas as pd
import nltk
from nltk import FreqDist #sklearn中应该也有相应的函数

# nltk.download('punkt')
corpus = 'this is my sentence'\
         'this is my life'\
         'this is the day'

tokens = nltk.word_tokenize(corpus)
#借用nltk的FreqDist统计文字出现的频率
fdist = FreqDist(tokens)
print(fdist['is'])
#可以把其中最常用的拿出来
#[('is', 3), ('my', 2), ('this', 1)]
standard_freq_vector = fdist.most_common(50)
size = len(standard_freq_vector)
print(standard_freq_vector)

#Func:按照出现频率的大小，记录下每一个单词的位置
def position_lookup(v):
    res = {}
    counter = 0
    for word in v:
        res[word[0]] = counter
        counter += 1
    return res
#把标准的单词位置记下来
standard_position_dict = position_lookup(standard_freq_vector)
print(standard_position_dict)

#这时如果有个新的句子
sentence = 'this is cool'
#新建一个跟我们标准vector同样大小的向量
freq_vector = [0] * size
#简单的对新句子处理下
tokens = nltk.word_tokenize(sentence)
for word in tokens:
    try:
        freq_vector[standard_position_dict[word]] += 1
    except KeyError:
        #如果是个新词，就Pass
        continue
print(freq_vector)






