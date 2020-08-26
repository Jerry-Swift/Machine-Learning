import numpy as np
import math
import random
import matplotlib.pyplot as plt
import sys
sys.path.append(r'D:\GitHub\Machine-Learning\TF2.0')
import 

data_path = 'Aggregation_cluster.txt'

def load_data():
    points = np.loadtxt(data_path, delimiter='\t')  #读取数据
    return points

def cal_dis(data, clu, k):  #计算所有的样本点分别与k个质心之间的距离
    dis = []
    for i in range(len(data)):
        dis.append([])
        for j in range(k):
            dis[i].append(math.sqrt((data[i, 0] - clu[j, 0])**2 + (data[i, 1]**2)))     #sqrt(x)返回x的平方根
    return np.asarray(dis)

def divide(data, dis):
    clusterRes = [0] * len(data)    #创建初始列表并将数据置零
    print(clusterRes)
    for i in range(len(data)):
        seq = np.argsort(dis[i])    #返回每个样本点对每个质心距离排序后的索引值
        clusterRes[i] = seq[0]      #将所有的样本点归类到不同的质心

    return np.asarray(clusterRes)

def center(data, clusterRes, k):    #
    clunew = []
    for i in range(k):
        print(np.where(clusterRes == i))
        idx = np.where(clusterRes == i)    #返回索引
        print(idx)
        sum = data[idx].sum(axis=0)
        avg_sum = sum/len(data[idx])
        clunew.append(avg_sum)
    clunew = np.asarray(clunew)
    return clunew[:, 0:2]

def classfy(data, clu, k):
    clulist = cal_dis(data, clu, k)
    clusterRes = divide(data, clulist)
    clunew = center(data, clusterRes, k)
    err = clunew - clu
    return err, clunew, k ,clusterRes

def plotRes(data, clusterRes, clusterNum):
    nPoints = len(data)
    scattersColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    for i in range(clusterNum):
        color = scattersColors[i % len(scattersColors)]
        x1 = []; y1=[]
        for j in range(nPoints):
            if clusterRes[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])
        plt.scatter(x1, y1, c=color, alpha=1, marker='+')
    plt.show()

if __name__ == '__main__':
    k = 7
    data = load_data()
    clu = random.sample(data[:, 0:2].tolist(), k)   #随机取k个初始质心 tolist()将numpy数组转换为list列表 data[:, 0:2]表示取所有行，第0到第1列的数据
    clu = np.asarray(clu)
    err, clunew, k, clusterRes = classfy(data, clu, k)
    while np.any(abs(err) > 0):
        print(clunew)
        err, clunew, k, clusterRes = classfy(data, clunew, k)

    clulist = cal_dis(data, clunew, k)
    clusterResult = divide(data, clulist)
    nmi, acc, purity = eva.eva(clusterResult, np.asarray(data[:, 2]))
    print(nmi, acc, purity)
    plotRes(data, clusterResult, k)

