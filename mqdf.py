#coding:utf-8

import numpy as np
from math import log
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets


class MQDF:
    def __init__(self, alpha=0.5, kk=1):
        self.alpha = alpha
        self.kk = kk
    
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
        cov = np.zeros([self.n_dim, self.n_dim])
        self.delta = 0
        self.eval_list = []
        self.evec_list = []
        for i in range(self.n_class):
            #クラスiのデータのインデックス
            index = np.where(label==i)
            imin = index[0].min()
            imax = index[0].max() + 1

            #クラスiのデータ取得
            data_part = data[imin:imax,:]

            #共分散行列
            #for j in range(data_part.shape[0]):
            #    tmp = (data_part[j] - mv_list[i])
            #    cov += tmp[:,np.newaxis] * tmp[np.newaxis,:]
            #cov /= float(data_part.shape[0])
            cov = np.cov(data_part.T) * (data_part.shape[0]-1) / data_part.shape[0]

            #固有値，固有ベクトル
            eval, evec = np.linalg.eig(cov)
            eval = eval[-1::-1] #降順
            self.eval_list.append(eval)
            evec = evec[:,-1::-1]
            self.evec_list.append(evec)

            #delta
            self.delta += np.sum(eval)
        self.delta /= (self.n_class * self.n_dim)

    def predict(self, X):
        y = np.zeros([X.shape[0], 1])
        sdvec = np.zeros(self.n_class)
        for i, x in enumerate(X):
            for j in range(self.n_class):
                xdd = x - self.mv_list[j]
                SD = np.dot(xdd, xdd)

                sh = self.alpha * self.delta

                DET = 0
                MBD5B = 0
                for k in range(self.kk):
                    DET += log((1-self.alpha) * self.eval_list[j][k] + sh)
                    P = np.dot(self.evec_list[j][:,k], xdd)
                    MBD5B += ((1-self.alpha) * self.eval_list[j][k]) / ((1-self.alpha) * self.eval_list[j][k] + sh) * P * P

                #MQDF
                sdvec[j] = (SD - MBD5B) / sh + DET

            y[i][0] = sdvec.argmin()
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

    #MQDF
    kk = 1
    alpha = 0.9
    clf = MQDF(kk=kk, alpha=alpha)
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
    plt.title("MQDF:(k=%d, alpha=%.1f)" % (kk, alpha))
    plt.show()
