#coding:utf-8

import numpy as np
from math import log
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets


class LDF:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
    
    def fit(self, data, label):
        #クラス数，次元数
        self.n_class = label.max() - label.min() + 1
        self.n_dim = data.shape[1]

        #平均ベクトル
        self.mv_list = []
        for i in range(self.n_class):
            #クラスiのデータのインデックス
            index = np.where(label==i)
            imin = index[0].min()
            imax = index[0].max() + 1

            #クラスiのデータ取得
            data_part = data[imin:imax,:]

            #平均ベクトル
            self.mv_list.append(data_part.mean(axis = 0))

        #共分散行列
        n_sample_list = []
        cov = np.zeros([self.n_dim, self.n_dim])
        cov_list = []
        for i in range(self.n_class):
            #クラスiのデータのインデックス
            index = np.where(label==i)
            imin = index[0].min()
            imax = index[0].max() + 1

            #サンプル数
            n_sample_list.append(len(index[0]))
            
            #クラスiのデータ取得
            data_part = data[imin:imax,:]

            #共分散行列
            cov = np.cov(data_part.T) * (data_part.shape[0]-1) / data_part.shape[0]
            cov_list.append(cov)

        #加重和
        sum = 0
        self.covw = np.zeros([self.n_dim, self.n_dim])
        for i, cov_tmp in enumerate(cov_list):
            self.covw += float(n_sample_list[i]) * cov_tmp
            sum += n_sample_list[i]
        self.covw /= float(sum)
        
        #正則化
        trace = np.trace(self.covw)
        identity = np.identity(self.n_dim)
        self.covw = (1-self.alpha) * self.covw
        self.covw += self.alpha * (trace / float(self.n_dim)) * identity
        
        #逆行列
        self.covw = np.linalg.inv(self.covw)

    def predict(self, X):
        y = np.zeros([X.shape[0], 1])
        sdvec = np.zeros(self.n_class)
        for i, x in enumerate(X):
            for j in range(self.n_class):
                sdvec[j] = np.dot( np.dot(self.mv_list[j][np.newaxis,:], self.covw), (2 * x[:,np.newaxis] - self.mv_list[j][:,np.newaxis]))
            y[i][0] = sdvec.argmax()
        return y

if __name__ == "__main__":
    #irisデータ
    iris = datasets.load_iris()

    #データ選択
    X = iris.data[:, :2]
    y = iris.target

    #カラーマップ
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    #LDF
    alpha = 0.5
    clf = LDF(alpha=alpha)
    clf.fit(X, y)

    #メッシュ作成
    h = .01 #メッシュのステップサイズ
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    #各位置(220x280)で予測
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
    #(61600,0)→(220,280)
    Z = Z.reshape(xx.shape)

    #figureオブジェクト
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    #プロット
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("LDF:(alpha=%.1f)" % (alpha))
    plt.show()
