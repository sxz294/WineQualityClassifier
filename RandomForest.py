import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import svm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=1)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

gamma_range = np.logspace(-7,2,10)
degree_range=np.arange(5,50,5)
accuracy_train=[]
accuracy_test=[]

depth_range=np.arange(5,200,5)
subtree=np.arange(50,200,10)
for n in subtree:
    clf=RandomForestClassifier(n_estimators=n,max_depth=35,random_state=0)
    clf.fit(X_train,y_train)
    y_h=clf.predict(X_test)
    accuracy_test.append(accuracy_score(y_test, y_h))
print(accuracy_test)