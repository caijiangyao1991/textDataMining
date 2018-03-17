#使用autoencoder降维以后再使用kmeans
#可以完全实现无监督，但是算法的准确性可能不高
from keras.layers import Input,Dense
from keras.models import Model
from sklearn.cluster import KMeans

class ASCIIAutoencoder():
    def __init__(self, sen_len=512, encoding_dim=32, epoch=50, val_ratio=0.3):
        """

        :param sen_len:把sentences pad成相同的长度, 保证输入等长才能学习
        :param encoding_dim:压缩后的维度dim
        :param epoch:要跑多少epoch
        :param val_ratio:
        """
        self.sen_len=sen_len
        self.encoding_dim=encoding_dim
        self.autoencoder = None
        self.encoder = None
        self.kmeanmodel = KMeans(n_cluster=2)
        self.epoch = epoch

    def preprocess(self,s_list,length=256):
        """
        把text扩展或者截断成length=512维，并且每个字符用ascil码表示，1，2,3,4,5,6等26个数字的表达形式
        要根据数据情况重新写，此次就不写了
        :param x:
        :return:
        """
        return s_list
    def fit(self, x):
        """
        模型构建
        :param x:input text
        :return:
        """
        #把所有的trainset都搞成同一个size，并把每一个字符都换成ascii码
        x_train = self.preprocess(x, length=self.sen_len)
        #给input预留好位置
        input_text = Input(shape=(self.sen_len,))
        #"encoded"每一经过一层，都被刷新成小一点的“压缩后表达式”
        encoded = Dense(1024, activation='tanh')(input_text)
        encoded = Dense(512, activation='tanh')(encoded)
        encoded = Dense(128, activation='tanh')(encoded)
        encoded = Dense(self.encoding_dim,activation='tanh')(encoded)
        #"decoded"就是把刚刚压缩完的东西，反过来还原成input_text
        decoded = Dense(128, activation='tanh')(encoded)
        decoded = Dense(512, activation='tanh')(decoded)
        decoded = Dense(1024, activation='tanh')(decoded)
        decoded = Dense(self.sen_len, activation='sigmoid')(decoded)

        #整个从大到小再到大的model叫autoencoder
        self.autoencoder = Model(input=input_text, output=decoded)
        #那么，只有从大到小（也就是一半的model）叫encoder
        self.encoder = Model(input=input_text, output=encoded)

        #同理，接下来我们搞一个decoder出来，也就是从小到大的model
        #首先encoder的input size预留好位置
        encoded_input = Input(shape=(1024,))
        #autoencoder的最后一层，就应该是decoder的最后一层
        decoder_layer = self.autoencoder.layers[-1]
        #然后从头到尾连起来，就是一个decoder
        decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

        #compile
        self.autoencoder.compile(optimizer='adam', loss='mse')
        #跑起来
        self.autoencoder.fit(x_train, x_train, nb_epoch=self.epoch, batch_size=1000,shuffle=True,)

        #这一部分是自己拿自己train一下KNN，一件简单的基于距离的分类器
        x_train = self.encoder.predict(x_train) #压缩成32位
        self.kmeanmodel.fit(x_train)

    def predict(self,x):
        """
        做预测
        :param x:input text
        :return:prediction
        """
        x_test = self.preprocess(x, length=self.sen_len)
        x_test = self.encoder(x_test)
        preds = self.kmeanmodel.predict(x_test)
        return preds


