import numpy as np
from pandas import read_csv
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
np.set_printoptions(precision=3)

# load wine quality data set
url = "https://raw.githubusercontent.com/sxz294/WineQualityClassifier/master/winequality-white.csv"
# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
wq = read_csv(url)
print(wq.head())
print(wq.groupby('"quality"').size())
# wq.plot(kind='box', subplots=True, layout=(1,4), sharex=False, sharey=False, figsize=(12,4))
# wq.hist()
# plt.show()

# # split data and apply standardize data
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=2)
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
# print(len(X_train))
# print(len(X_test))


# https://github.com/sxz294/WineQualityClassifier/blob/master/winequality-white.csv