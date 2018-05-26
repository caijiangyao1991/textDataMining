#encoding=utf-8
"""
kaggle:https://www.kaggle.com/c/home-depot-product-search-relevance
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


#TODO 读入数据
df_train = pd.read_csv("./data/homepeot/train.csv",encoding="ISO-8859-1")
df_test = pd.read_csv("./data/homepeot/test.csv",encoding="ISO-8859-1")
df_desc = pd.read_csv("./data/homepeot/product_descriptions.csv")

#合并测试/训练集，以便统一做进一步的文本预处理
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True) #上下粘起来
print(df_all.shape)
#产品介绍也是一个极有用的信息，我们把它拿过来
df_all = pd.merge(df_all, df_desc, how='left',on='product_uid')
print(df_all.head())

#TODO 文本预处理
#这里遇到的文本预处理比较简单，因为主要就是看关键词是否会被包含，
# 所以我们统一化我们的文本内容，以达到任何term在我们数据集中只有一种表达式的效果
# 这里我们用简单的Stem做个例子（有兴趣的还可以去掉停用词，纠正拼音，去掉数字，去掉emoji等等）
stemmer = SnowballStemmer('english')
def str_stemmer(s):
    return " ".join([stemmer.stem(word) for word in s.lower().split()])
#为了计算“关键词”的有效性，我们可以naive地直接看【出现了多少次】
def str_common_word(str1, str2):
    return sum(int(str2.find(word) for word in str1.split()))

#接下来把每一个column都跑一遍，以清洁所有的文本内容
df_all["search_term"] = df_all["search_term"].map(lambda x: str_stemmer(x))
df_all["product_title"] = df_all["product_title"].map(lambda x: str_stemmer(x))
df_all["product_description"] = df_all["product_description"].map(lambda x: str_stemmer(x))

#TODO 自制文本特征
#脑洞打开的过程，当然也不是想加什么就加什么
#关键词的长度
df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
#描述中有多少关键词重合
df_all['commons_in_title'] = df_all.apply(lambda x:str_common_word(x['search_term'],x['product_title']), axis=1)
df_all['commons_in_desc'] = df_all.apply(lambda x:str_common_word(x['search_term'],x['product_description']), axis=1)

#把不能被机器学习模型处理的column给处理掉
df_all = df_all.drop(['search_term','product_title','product_description'],axis=1)

#TODO 重塑训练/测试集
df_train = df_all.loc[df_train.index]
df_test = df_all.loc[df_test.index]
#记录下测试集的id,留着上传时候能对上号
test_ids = df_test['id']
#分离出y_train
y_train = df_train['relevance'].values
#把原集合中的label删除
x_train = df_train.drop(['id','relevance'], axis=1).values
x_test = df_test.drop(['id','relevance'], axis=1).values

#TODO 建立模型
#用cv结果保证公正客观，调试不同alpha
params = [1, 3, 5, 6, 7, 8, 9, 10]
test_scores = []
for param in params:
    clf = RandomForestRegressor(n_estimators=30, max_depth=params)
    test_scores = np.sqrt(-cross_val_score(clf, x_train, y_train, cv=5, scoring= 'neg_mean_squared_error'))
    test_scores.append(np.mean(test_scores))

plt.plot(params, test_scores)
plt.title("params vs CV Error")
plt.show()

#上传结果
rf = RandomForestRegressor(n_estimators=30, max_depth=6)
rf.fit(x_train,y_train)
y_pred = rf.predict(x_test)
pd.DataFrame({"id":test_ids,"relevance":y_pred}).to_csv("./data/homepeote/submission.csv",index=False)