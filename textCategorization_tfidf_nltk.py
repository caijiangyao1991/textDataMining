#encoding=utf-8
import pandas as pd
import nltk
from nltk.text import TextCollection
from nltk import FreqDist

#TODO TFIDF
#term frequency * inverse Document Frequency
#把所有的文档放到TextCollection类中，这个类会自动帮你断句，做统计，做计算
corpus = TextCollection(['this is sentence one','this is sentence two','this is sentence three'])
#直接就能计算出tfidf
#(term:一句话中的某个term, text:这句话）
print (corpus.tf_idf('is','this is sentence one'))

#把预料中出现过的常用的单词做一个词汇表
tokens = nltk.word_tokenize('this is sentence one','this is sentence two' ,'this is sentence three')
fdist = FreqDist(tokens)
standard_freq_vector = fdist.most_common(50)
print(standard_freq_vector)
standard_vocab = []
for i in standard_freq_vector:
    standard_vocab.append(i[0])

#对于每个新句子，构造相同长度的词向量，计算我们词汇表里面的所有词在新句子中的tfidf,
#出现这个单词，则这个单词下面就有数
#对于你所有的句子而言，这个vacabulary是确定的，再有新句子进来也是拟合你之前语料库里面的词
new_sentence = 'this is sentence five'
for word in standard_vocab:
    print(corpus.tf_idf(word, new_sentence))

"""如果文本数据太大，把它存在硬盘里面，然后用iterater一行行读进来"""