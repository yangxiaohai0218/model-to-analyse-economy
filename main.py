import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 先对财政数据进行大体分析
def Analyse(file):
    outfile = "data/out1.csv"
    data = pd.read_csv(file)
    #print(data)
    # print(type(data))
    #观察最大值，最小值，平均值以及标准值
    result = [data.max(), data.min(), data.mean(), data.std()]
    result = pd.DataFrame(result, index=["max", "min", "mean", "std"])
    result.plot(kind='bar')
    import matplotlib.pyplot as plt
    #plt.show()
    #np.round(result, 4).to_csv(outfile)
    #观察其pearson相关性
    #np.round(data.corr(method='pearson'), 2).to_csv('data/out2.csv')
    #热力图分析
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.subplots(figsize=(16, 9))
    correlation_mat = data.corr()
    sns.heatmap(correlation_mat, annot=True, cbar=True, square=True, fmt='.2f', annot_kws={'size': 5})

   # plt.show()
    return data

# lasso回归模型
def lasso(data):
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import LassoCV
    model=LassoCV() # 尝试利用LASSOCV ，lasso会报没有收敛的错
    model = Lasso()  # 正则化参数的选取很重要
    model.fit(data.iloc[:, :13], data['y'])
    print(model.coef_)
    # print(model.intercept_)
    result=pd.DataFrame(model.coef_,index=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13'])

   # np.round(result,5).to_csv("data\out4.csv")
   # np.round(result,4).to_csv("data/out4.csv")
    print(np.round(result,4))
    mark = model.coef_ != 0
    return model.coef_, np.array(mark)
# 自定义构建GM(1,1)模型：
def GM11(x0):
    # 数据检验,检验数列是否能够使用GM(1,1)模型
    # 计算级比
    x0 = np.array(x0)
    lens = len(x0)
    lambds = []
    for i in range(1, lens):
        lambds.append(x0[i - 1] / x0[i])
    # 计算区间
    X_min = np.e ** (-2 / (lens + 1))
    X_max = np.e ** (2 / (lens + 1))
    # 检验
    is_ok = True
    for lambd in lambds:
        if (lambd < X_min or lambd > X_max):
            is_ok = False
    if (is_ok == False):
        print('数据未通过检验')
        # return
    # GM(1,1)采用1-AGO序列
    x1 = x0.cumsum()
    z1 = (x1[:len(x1) - 1] + x1[1:]) / 2.0  # 紧邻均值（MEAN）生成序列 ,由常微分方程可知，取前后两个时刻的值的平均值代替更为合理
    # x0[1] = -1/2.0*(x1[1] + x1[0])
    z1 = z1.reshape((len(z1), 1))
    B = np.append(-z1, np.ones_like(z1), axis=1)
    Yn = x0[1:].reshape((len(x0) - 1, 1))
    [[a], [b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Yn)  # 计算参数
    # fkplusone = (x1[0]-b/a)*np.exp(-a*k)#时间响应方程 # 由于x0[0] = x1[0]
    f = lambda k: (x1[0] - b / a) * np.exp(-a * (k - 1)) - (x1[0] - b / a) * np.exp(-a * (k - 2))  # 还原值
    delta = np.abs(x0 - np.array([f(i) for i in range(1, len(x0) + 1)]))  # 残差
    C = delta.std() / x0.std()  # 后验比差值
    P = 1.0 * (np.abs(delta - delta.mean()) < 0.6745 * x0.std()).sum() / len(x0)
    return f, a, b, C, P  # 返回灰色预测函数、a、b、后验差之比、小残差概率

# 利用GM(1,1)模型进行预测：
def PredictbyGM11(data):
    result = lasso(data)
    mark = result[1]
    #print(mark)
    label = np.array(data.columns)
    data.index = range(1994, 2014)  # 数据有20行
    data.loc[2014] = None
    data.loc[2015] = None
    Truelabel = []
    for i in range(len(mark)):
        if mark[i] == True:
            Truelabel.append(label[i])
    Truelabel = np.array(Truelabel)
    # print(Truelabel)
    for i in Truelabel:
        gm = GM11(data[i][:-2].values)
        f = gm[0]
        p = gm[-1]
        c = gm[-2]
        data[i][2014] = f(len(data) - 1)
        data[i][2015] = f(len(data))
        data[i] = np.round(data[i], 2)
        assess = []  # 用来保存评价结果,其中0表示结果很好，1表示合格，2是勉强合格，3是不合格
        if (c < 0.35 and p > 0.95):
            assess.append(0)
        elif (c < 0.5 and p > 0.8):
            assess.append(1)
        elif (c < 0.65 and p > 0.7):
            assess.append(2)
        else:
            assess.append(3)
    return data, assess, Truelabel  #返回更改的数据以及特征

# 构建神经网络模型：
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
def creatmodel(num):
    model = Sequential()
    model.add(Input(num, ))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(1))
    return model

# 训练神经网络模型，并预测结果，并画出预测图像
#import matplotlib.pyplot as plt  #
from matplotlib import pyplot as plt
def train(data, Truelabel):
    # 数据预处理
    newdata=data.copy(deep=True)
    data_train = data[:][:-2]
    # data_train = data.loc[range(1999, 2014)].copy()
    data_mean = data_train.mean()
    data_std = data_train.std()
    train_data = (data_train - data_mean) / data_std
    # print(train_data)
    x_train = train_data[Truelabel].values
    y_train = train_data["y"].values
    # print(x_train)
    # print(y_train)
    model = creatmodel(len(Truelabel))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(x_train, y_train, epochs=3000,batch_size=16)
    # x = ((data[Truelabel] - data_mean[Truelabel]) / data_std[Truelabel]).as_matrix()
    # data[u'y_pred'] = model.predict(x) * data_std['y'] + data_mean['y']
    # data[u'y_pred'] = data[u'y_pred'].round(2)
    x = ((newdata[Truelabel] - data_mean[Truelabel]) / data_std[Truelabel]).values
    #y_std=data_train['y'].std()
    #y_mean=data_train['y'].mean()
    newdata['y_pred'] = model.predict(x) * newdata['y'].std()+newdata['y'].mean()
    newdata['y_pred'] = newdata['y_pred'].round(2)
    print(newdata[["y","y_pred"]])
    p=newdata[['y', 'y_pred']].plot(subplots=True, style=['b-o', 'r-*'])
   # plt.show()
    return newdata

#构建linearSVR模型，并预测结果，返回预测图像
from sklearn.svm import LinearSVR
def LinearSVRmodel(data,Truelabel):
    newdata=data.copy(deep=True)
    train_data=data[:][:-2]
    data_mean=train_data.mean()
    data_std=train_data.std()
    data_train=(train_data-data_mean)/data_std
    x_train=data_train[Truelabel].values
    y_train=data_train['y'].values
    x = ((data[Truelabel] - data_mean[Truelabel]) / data_std[Truelabel]).values
    linearsvr = LinearSVR().fit(x_train, y_train)
    newdata[u'y_pred'] = linearsvr.predict(x) * data_std['y'] + data_mean['y']
   # data.to_excel("data/linearAVR")
    print(newdata[["y","y_pred"]])
    newdata[['y', 'y_pred']].plot(subplots=True, style=['b-o', 'r-*'], xticks=newdata.index[::2])
   # plt.show()
    return newdata
from pyGRNN import GRNN
from sklearn.model_selection import  GridSearchCV
def GrnnModel(data,Truelabel):
    newdata = data.copy(deep=True)
    train_data = data[:][:-2]
    data_mean = train_data.mean()
    data_std = train_data.std()
    data_train = (train_data - data_mean) / data_std
    x_train = data_train[Truelabel].values
    y_train = data_train['y'].values
    x = ((data[Truelabel] - data_mean[Truelabel]) / data_std[Truelabel]).values
    IGRNN=GRNN()
    params_IGRNN = {'kernel': ["RBF"],
                    'sigma': list(np.arange(0.1, 4, 0.01)),
                    'calibration': ['None']
                    }
    grid_IGRNN = GridSearchCV(estimator=IGRNN,
                              param_grid=params_IGRNN,
                              scoring='neg_mean_squared_error',
                              cv=5,
                              verbose=1
                              )
    grid_IGRNN.fit(x_train, y_train)
    best_model = grid_IGRNN.best_estimator_
   # y_pred = best_model.predict(x_train)
    newdata[u'y_pred'] = best_model.predict(x) * data_std['y'] + data_mean['y']
    #print(newdata[["y", "y_pred"]])
    newdata[['y', 'y_pred']].plot(subplots=True, style=['b-o', 'r-*'], xticks=newdata.index[::2])
    #plt.show()
    return newdata



#对比神经网络模型和linearSVR模型
from sklearn.metrics import explained_variance_score,\
mean_absolute_error,mean_squared_error,\
median_absolute_error,r2_score
def contrast(data1,data2,data3):
    data1=data1[:-2]
    data2=data2[:-2]
    data3=data3[:-2]
    model1=[]
    model2=[]
    model3=[]
    model1.append(mean_absolute_error(data1['y'].values, data1['y_pred'].values))
    model1.append(mean_squared_error(data1['y'], data1['y_pred']))#MSE的值越小，说明预测模型描述实验数据具有更好的精确度
    model1.append(median_absolute_error(data1['y'], data1['y_pred']))
    model1.append(explained_variance_score(data1['y'], data1['y_pred']))
    model1.append(r2_score(data1['y'], data1['y_pred']))#表明方程的变量对y的解释能力越强，这个模型对数据拟合的也较好

    model2.append(mean_absolute_error(data2['y'].values, data2['y_pred'].values))
    model2.append(mean_squared_error(data2['y'], data2['y_pred']))
    model2.append(median_absolute_error(data2['y'], data2['y_pred']))
    model2.append(explained_variance_score(data2['y'], data2['y_pred']))
    model2.append(r2_score(data2['y'], data2['y_pred']))

    model3.append(mean_absolute_error(data3['y'].values, data3['y_pred'].values))
    model3.append(mean_squared_error(data3['y'], data3['y_pred']))
    model3.append(median_absolute_error(data3['y'], data3['y_pred']))
    model3.append(explained_variance_score(data3['y'], data3['y_pred']))
    model3.append(r2_score(data3['y'], data3['y_pred']))

    return model1,model2,model3

#构造主函数
def main():
    data = Analyse("data/data1.csv")
    result = PredictbyGM11(data)
    #print(result[0])
    data1=train(result[0], result[2])
    data2= LinearSVRmodel(result[0], result[2])
    data3 = GrnnModel(result[0], result[2])
    model=contrast(data1,data2,data3)
   # print(result[1])
    print(model[0])
    print(model[1])
    print(model[2])


#运行程序
if __name__ == "__main__":
    main()






