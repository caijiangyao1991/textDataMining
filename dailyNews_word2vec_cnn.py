#用每日新闻预测金融市场变化
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from datetime import date

#其实看起来特别的简单直观。如果是1，那么当日的DJIA就提高或者不变了。如果是0，那么DJIA那天就是跌了
data = pd.read_csv('./data/Combined_News_DJIA.csv')
#TODO 分割测试/训练集
#这下，我们可以先把数据给分成Training/Testing data
train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']
#我们把每条新闻做成一个单独的句子，集合在一起
#corpus是全部我们『可见』的文本资料。我们假设每条新闻就是一句话，把他们全部flatten()了，我们就会得到list of sentences。
#同时我们的X_train和X_test可不能随便flatten，他们需要与y_train和y_test对应。
X_train = train[train.columns[2:]]
corpus = X_train.values.flatten().astype(str)
X_train = X_train.values.astype(str)
X_train = np.array([' '.join(x) for x in X_train]) #是拍平，但是每列合到一起，代表一个元素，与

X_test = test[test.columns[2:]]
X_test = X_test.values.astype(str)
X_test = np.array([' '.join(x) for x in X_test])

y_train = train['Label'].values  #用values会变成一个list
y_test = test['Label'].values
#来，我们再把每个单词给分隔开：同样，corpus和X_train的处理不同
#我们可以看到，虽然corpus和x都是一个二维数组，但是他们的意义不同了。
#corpus里，第二维数据是一个个句子。x里，第二维数据是一个个数据点（对应每个label）
from nltk.tokenize import word_tokenize
corpus = [word_tokenize(x) for x in corpus]
X_train = [word_tokenize(x) for x in X_train]
X_test = [word_tokenize(x) for x in X_test]

#TODO 预处理把我们的文本资料变得更加统一
#最小化， 删除停用词，删除数字与符号
#lemma

#停用词
from nltk.corpus import stopwords
stop = stopwords.words('english')
#数字
import re
def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))
#特殊符号
def isSymbol(inputString):
    return bool(re.match(r'[^\w]', inputString))
#lemma
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
def check(word):
    """如果需要这个单词，则True,如果应该去除，则False"""
    word = word.lower()
    if word in stop:
        return False
    elif hasNumbers(word) or isSymbol(word):
        return False
    else:
        return True

#把上面方法综合起来
def preprocessing(sen):
    res = []
    for word in sen:
        if check(word):
            word = word.lower().replace("b'",'').replace('b"','').replace('"','').replace("'",'')
            res.append(word)
    return res

corpus = [preprocessing(x) for x in corpus]
X_train = [preprocessing(x) for x in X_train]
X_test = [preprocessing(x) for x in X_test]

#TODO 训练NLP模型
#有了这些干净的数据集，我们可以做我们的NLP模型了。我们先用最简单的Word2Vec
from gensim.models.word2vec import Word2Vec
model = Word2Vec(corpus, size=128, window=5, min_count=5, workers=4)
#这时候，每个单词都可以像查找字典一样，读出他们的w2v坐标了
#接着，我们于是就可以用这个坐标，来表示我们的之前干干净净的X。
#但是这儿有个问题。我们的vec是基于每个单词的，怎么办呢？由于我们文本本身的量很小，我们可以把所有的单词的vector拿过来取个平均值：

#先拿到全部的vocabulary
vocab = model.wv.vocab

#得到任意text的vector
def get_vector(wordlist):
    #建立一个全是0的array
    res = np.zeros([128])
    count = 0
    for word in wordlist:
        if word in vocab:
            res += model[word]
            count += 1
    return res/count

#TODO 这样，我们可以同步把我们的X都给转化成128维的一个vector list


#TODO 方法1：通过get_vector把每个单词的vector求平均，作为每个样本的vector
wordlist_train = X_train
wordlist_test = X_test
X_train = [get_vector(x) for x in X_train]
X_test = [get_vector(x) for x in X_test]
#TODO 建立ML模型
# 这里，因为我们128维的每一个值都是连续关系的。不是分裂开考虑的。
# 所以，道理上讲，我们是不太适合用RandomForest这类把每个column当做单独的variable来看的方法。当然，事实是，你也可以这么用
# 我们来看看比较适合连续函数的方法：SVM
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
params = [0.1, 0.5, 1, 3, 5, 7, 10, 12, 16, 20, 25, 30, 35, 40]
test_scores = []
for param in params:
    clf = SVR(gamma=params)
    test_score = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')
    test_scores.append(np.mean(test_score))

#TODO 查看最佳的参数
import matplotlib.pyplot as plt
plt.plot(params, test_scores)
plt.title("Params vs CV AUC Score")

#TODO 方法2：用CNN来提升逼格
#有些同学也许会说，这也太扯了吧，直接平均了一下vec值，就能拿来跑？我们必然是有更严谨的处理方式。
#比如：用vector表示出一个大matrix，并用CNN做“降维+注意力”
#（为了演示的方便，下面内容我会把整个case搞得简单点。要是想更加复杂准确的话，直接调整参数，往大了调，就行）
#首先，我们确定一个padding_size。什么是padding size？就是为了让我们生成的matrix是一样的size啊。。（具体见课件）
#（这里其实我们可以最简单地调用keras的sequence方法来做，但是我想让大家更清楚的明白一下，内部发生了什么）

# 说明，对于每天的新闻，我们会考虑前256个单词。不够256个单词的我们用[000000]128维（每一个都对应128维）补上
# vec_size 指的是我们本身vector的size
def transform_to_matrix(x, padding_size=256, vec_size=128):
    res = []
    for sen in x:
        matrix = []
        for i in range(padding_size):
            try:
                matrix.append(model[sen[i]].tolist()) #把每个单词对应的128维都添加进去
            except:
                matrix.append([0] * vec_size)
        res.append(matrix)
    return res


#这时候，我们把我们原本的word list跑一遍：
X_train = transform_to_matrix(wordlist_train)
X_test = transform_to_matrix(wordlist_test)
#可以看到，现在我们得到的就是一个大大的Matrix，它的size是 128 * 256,每一个这样的matrix，就是对应了我们每一个数据点

#搞成np的数组便于处理
X_train = np.array(X_train)
X_test = np.array(X_test)
print(X_train[123])
#查看数组大小
print (X_train.shape) #(1611, 256, 128)
print (X_test.shape) #(378, 256, 128)

##在进行下一步之前，我们把我们的input要reshape一下。
#原因是我们要让每一个matrix外部“包裹”一层维度。来告诉我们的CNN model，我们的每个数据点都是独立的。之间木有前后关系。
#（其实对于股票这个case，前后关系是存在的。这里不想深究太多这个问题。有兴趣的同学可以谷歌CNN+LSTM这种强大带记忆的深度学习模型。）
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])

#TODO 接下来，我们安安稳稳的定义我们的CNN模型
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten

#set parameters
batch_size = 32
n_filter = 16
filter_length = 4
nb_epoch = 5
n_pool = 2

#新建一个sequential的模型
model = Sequential()
model.add(Convolution2D(n_filter, filter_length, filter_length, input_shape=(1, 256, 128)))
model.add(Activation('relu'))
model.add(Convolution2D(n_filter, filter_length, filter_length))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(n_pool, n_pool)))
model.add(Dropout(0.25))
model.add(Flatten())
#后面接上一个ANN
## 我们这里使用了最简单的神经网络~ 你可以用LSTM或者RNN等接在CNN的那句Flatten之后
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('softmax'))
#complile模型
model.compile(loss='mse',optimizer='adadelta',metrics=['accuracy'])
#可以放进去我们的X和Y了
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
score = model.evaluation(X_train, y_test, verbose=0)
print('Test score:',score[0])
print('Test accuracy:',score[1])

"""思考：
虽然我们这里使用了word2vec，但是对CNN而言，管你3721的input是什么，只要符合规矩，它都可以process。
这其实就给了我们更多的“发散”空间：
我们可以用ASCII码（0，256）来表达每个位置上的字符，并组合成一个大大的matrix。
这样牛逼之处在于，你都不需要用preprocessing了，因为每一个字符的可以被表示出来，并且都有意义。
另外，你也可以使用不同的分类器：
我们这里使用了最简单的神经网络~
你可以用LSTM或者RNN等接在CNN的那句Flatten之后"""





