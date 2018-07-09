#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

#irisデータ
iris = datasets.load_iris()

#データ選択
X = iris.data[:, :2]
y = iris.target

#カラーマップ
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
#k近傍法
n_neighbors = 15
weights = "uniform"
clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
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
plt.title("K-Nearest Neighbors:(K = %i)" % (n_neighbors))
plt.show()
