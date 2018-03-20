import os
import numpy as np
import nltk
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from gensim.models.word2vec import Word2Vec

raw_text = ''
for file in os.listdir("./data/"):
    if file.endswith("-0.txt"):
        raw_text += open("./data/" + file, errors='ignore').read() + '\n\n'
raw_text = raw_text.lower()
sentensor = nltk.data.load('tokenizers/punkt/english.pickle') #将手上的文本，分成一个个句子
sents = sentensor.tokenize(raw_text)
corpus = []
for sen in sents:
    corpus.append(nltk.word_tokenize(sen))  #再将句子分成一个个词例如[['\ufeffthe', 'project', 'gutenberg', 'ebook', '.'], ['you', 'may', 'copy']]

#好，word2vec乱炖
#在这模型中 可以加上其他语料 ， 因为真正训练是用lstm，w2v不会影响，
#对每个词进行稀疏编码，总共对14050个词进行了稀疏编码（词频小于5的单词会被丢弃），另外表达的维度是128维。
w2v_model = Word2Vec(corpus, size=128, window=5, min_count=5, workers=4) #是对每一个词进行了编码？总共出现14050个词
# print(w2v_model)
raw_input = [item for sublist in corpus for item in sublist ] #1784533

text_stream = []
vocab = w2v_model.wv.vocab
for word in raw_input:
    if word in vocab:
        text_stream.append(word) #把稀疏编码的词留下，去掉了低频的那些词

#我们这里文本预测就是，给了前面的单词以后，下面一个单词是谁？比如，hello from the other, 给出 side
#构造训练集
#我们需要把我们的raw text变成可以用来训练的x,y:
seq_length = 10
x = []
y = []
for i in range(0,len(text_stream)-seq_length):
    given = text_stream[i:i+seq_length]
    predict = text_stream[i + seq_length]
    x.append(np.array(w2v_model[word] for word in given))
    y.append(w2v_model[predict])

#我们已经有了一个input的数字表达（w2v），我们要把它变成LSTM需要的数组格式： [样本数，时间步伐，特征]
#第二，对于output，我们直接用128维的输出
x = np.reshape(x, (-1, seq_length, 128))
y = np.reshape(y, (-1,128))

#模型构建
model = Sequential()
model.add(LSTM(256, dropout_W=0.2, dropout_U=0.2, input_shape=(seq_length, 128)))
model.add(Dropout(0.2))
model.add(Dense(128, activation='sigmoid'))
model.compile(loss='mse',optimizer='adam')

#跑模型
model.fit(x, y, np_epoch=50, batch_size=4096)

#看看我们训练的LSTM的效果
def predict_next(input_array):
    x = np.reshape(input_array,(-1, seq_length, 128))
    y = model.predict(x)
    return y
def string_to_index(raw_input):
    raw_input = raw_input.lower()
    input_stream = nltk.word_tokenize(raw_input)
    res = []
    for word in input_stream[len(input_stream)-seq_length:]:
        res.append(w2v_model[word])
    return res
def y_to_word(y):
    word = w2v_model.most_similar(positive=y, topn=1)
    return word

def generate_article(init, rounds=30):
    in_string = init.lower()
    for i in range(rounds):
        n = y_to_word(predict_next(string_to_index(in_string)))
        in_string += ' ' + n[0][0]
    return in_string

init = 'Language Models allow us to measure how likely a sentence is, which is an important for Machine'
article = generate_article(init)
print(article)






