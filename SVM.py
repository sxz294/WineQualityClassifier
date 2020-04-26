import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import svm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns',None)

# load wine quality data set
url = "https://raw.githubusercontent.com/sxz294/WineQualityClassifier/master/winequality-white.csv"
df = read_csv(url,sep=';')

# data visualization
# print(df.head())
# print(df.describe())
# print(df.groupby('quality').size())
# print(df.corr())
# df.plot(kind='box', subplots=True, layout=(1,12), sharex=False, sharey=False, figsize=(12,4))
# df.hist()
# from pandas.plotting import scatter_matrix
# scatter_matrix(df)
# plt.show()

data=df.values
col=df.columns.size
X=data[:,:col-1]
y=data[:,col-1]
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# X = SelectKBest(chi2, k=7).fit_transform(X, y)

# split data and apply standardize data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

gamma_range = np.logspace(-4,4,10)
degree_range=np.arange(5,50,5)
accuracy_train=[]
accuracy_test=[]
from sklearn.utils import class_weight
# class_weights = class_weight.compute_class_weight('balanced',
#                                                  np.unique(y_train),
#                                                  y_train)
# class_weight_dict = dict(enumerate(class_weight))

class_weights = dict(zip(np.unique(y_train), class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)))
for gamma in gamma_range:
    clf = svm.SVC(kernel='rbf', gamma=gamma,decision_function_shape='ovo',class_weight=class_weights)
    # clf = svm.SVC(kernel='rbf', gamma=gamma, decision_function_shape='ovo', class_weight=class_weights)
    # clf = svm.SVC(kernel='poly', degree=degree, decision_function_shape='ovo',class_weight=class_weights)
    c = clf.fit(X_train, y_train)
    accuracy_train.append(clf.score(X_train,y_train))
    y_h = c.predict(X_test)
    print(y_h)
    accuracy_test.append(c.score(X_test,y_test))
print(accuracy_train)
print(accuracy_test)
plt.semilogx(gamma_range,accuracy_train,label='training accuracy')
plt.semilogx(gamma_range,accuracy_test,label='test accuracy')
# plt(degree_range,accuracy_train,label='training accuracy')
# plt(degree_range,accuracy_test,label='test accuracy')
plt.xlabel("gamma")
plt.ylabel("accuracy")
plt.legend()
plt.title("Accuracy curve of training data and test data")
plt.show()
