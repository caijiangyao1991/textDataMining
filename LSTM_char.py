#我们这里用温斯顿丘吉尔的人物传记作为我们的学习语料。
#由前几个字母自动补全后一个字母
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

#我们把文本读入
raw_text = open('../input/Winston_Churchil.txt').read()
raw_text = raw_text.lower()
#既然我们是以每个字母为层级，字母总共才26个，
# 所以我们可以很方便的用One-Hot来编码出所有的字母，每个字母对应相应的index（当然，可能还有些标点符号和其他noise）
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
#我们这里简单的文本预测就是，给了前置的字母以后，下一个字母是谁？
#比如，Winsto, 给出 n Britai 给出 n

#构造训练测试集
#把原来的raw text变成可以用来训练的x, y
#x 是前置字母们 y 是后一个字母
seq_length = 100
x = []
y = []
for i in range(0, len(raw_text) - seq_length):
    given = raw_text[i:i + seq_length]
    predict = raw_text[i + seq_length]
    x.append([char_to_int[char] for char in given])
    y.append(char_to_int[predict])

#我们已经有了一个input的数字表达（index），我们要把它变成LSTM需要的数组格式： [样本数，时间步伐，特征]
#第二，对于output，用one-hot做output的预测可以给我们更好的效果，相对于直接预测一个准确的y数值的话。
#不再用算出一个精确的值 1还是2,3 还是4，这样的话scale太大了不太适合机器学习，One-hot后表达成向量形式，向量之间可以进行差值的比较，
#比如正确的y可能是[0,0,1,0,0,0,0,0,0]而我们预测出的可能是[0.1,0.21,0.9,0.03,0.21,0.09]这里面只要输出0.9这个最高可能性的值就可以了，
#加大了模型的容错率
n_patterns = len(x)
n_vocab = len(chars)
# 把x变成LSTM需要的样子
x = numpy.reshape(x, (n_patterns, seq_length, 1))
# 简单normal到0-1之间
x = x / float(n_vocab)
# output变成one-hot
y = np_utils.to_categorical(y)

#模型建造
#LSTM模型构建
model = Sequential()
model.add(LSTM(256, input_shape=(x.shape[1], x.shape[2]))) #256是layer的数量 数量越大精确度越高
model.add(Dropout(0.2))
model.add(Dense(y.shape[1],activation='softmax')) #普通神经网络称为dense
model.compile(loss='categorical_crossentropy', optimizer='adam')
#跑模型
model.fit(x, y, nb_epoch=50, batch_size=4096)

#我们来写个程序，看看我们训练出来的LSTM的效果：
def predict_next(input_array):
    x = numpy.reshape(input_array, (1, seq_length, 1))
    x = x/float(n_vocab)
    y = model.predict(x)
    return y
def string_to_index(raw_input):
    res = []
    for c in raw_input[(len(raw_input)-seq_length):]: #预测后面那100个字母后的单词
        res.append(char_to_int[c])
    return res
def y_to_char(y):
    largest_index = y.argmax()
    c = int_to_char[largest_index]
    return c
#好，写一个大程序 跑两百次，相当于预测后面200个词，每次新的词又加入，每次预测后一个词，这样可以想补全多少个就补全多少个
def generate_article(init, rounds=200):
    in_string = init.lower()
    for i in range(rounds):
        n = y_to_char(predict_next(string_to_index(in_string)))
        in_string += n
    return in_string

#预测开始
init = 'His object in coming to New York was to engage officers for that service. He came at an opportune moment'
article = generate_article(init)
print(article)



