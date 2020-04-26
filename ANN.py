import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns',None)

# load wine quality data set
url = "https://raw.githubusercontent.com/sxz294/WineQualityClassifier/master/winequality-white.csv"
df = read_csv(url,sep=';')

# data visualization
print(df.head())
print(df.describe())
print(df.groupby('quality').size())
print(df.corr())
# df.plot(kind='box', subplots=True, layout=(1,12), sharex=False, sharey=False, figsize=(12,4))
# df.hist()
# from pandas.plotting import scatter_matrix
# scatter_matrix(df)
# plt.show()

data=df.values
col=df.columns.size
X=data[:,:col-1]
y=data[:,col-1]
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X = SelectKBest(chi2, k=7).fit_transform(X, y)

print(X)
print(y)

# split data and apply standardize data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=2)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# print(len(X_train))
# print(len(X_test))




# train the model and predict test set
accuracy_train=[]
accuracy_test=[]
hls=np.arange(50,1000,50)
for size in hls:
    clf=MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(size,),max_iter=5000, random_state=1)
    # clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(n,), learning_rate='constant',
    #        learning_rate_init=1e-5,max_iter=10000,random_state=1)
    clf.fit(X_train, y_train)
    accuracy_train.append(clf.score(X_train, y_train))
    accuracy_test.append(clf.score(X_test, y_test))
# print(accuracy_train)
# print(accuracy_test)
optSize=(np.argmax(accuracy_test)+1)*50
bestAccuracy=max(accuracy_test)
plt.plot(hls,accuracy_train,label='training accuracy')
plt.plot(hls,accuracy_test,label='test accuracy')
plt.plot(hls,np.ones(len(hls))*bestAccuracy,'r--')
plt.xlabel("hidden layer size")
plt.ylabel("accuracy")
plt.text(optSize,bestAccuracy+0.005,"optimal hidden layer size is %s" % optSize)
plt.legend()
plt.title("Accuracy curve of training data and test data")
plt.show()

