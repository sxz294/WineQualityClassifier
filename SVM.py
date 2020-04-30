import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns',None)

# load wine quality data set
url = "https://raw.githubusercontent.com/sxz294/WineQualityClassifier/master/winequality-white.csv"
df = read_csv(url,sep=';')

# data visualization
fmt = '%.2f'
yticks = mtick.FormatStrFormatter(fmt)
print(df.head())
print(df.describe())
print(df.groupby('quality').size())
print(df.corr())
ax1=df.plot(kind='box', subplots=True, layout=(2,6), sharex=False, sharey=False, figsize=(12,6))
for ax in ax1:
    ax.yaxis.set_major_formatter(yticks)
plt.tight_layout()

# detect and remove outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df_out = df[~((df < (Q1 - 3 * IQR)) |(df > (Q3 + 3 * IQR))).any(axis=1)]
print(((df < (Q1 - 3 * IQR)) |(df > (Q3 + 3 * IQR))).any(axis=1))
df_out.shape
print(df_out.groupby('quality').size())
ax2=df_out.plot(kind='box', subplots=True, layout=(2,6), sharex=False, sharey=False, figsize=(12,6))
for ax in ax2:
    ax.yaxis.set_major_formatter(yticks)
plt.tight_layout()
plt.show()

data=df_out.values
col=df_out.columns.size
X=data[:,:col-1]
y=data[:,col-1]

# split data and standardize data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=1)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)




# train model
start_time=time.time()
pipeline = make_pipeline(SVC(kernel='rbf',random_state=0,decision_function_shape='ovr'))
hyperparameters = { 'svc__gamma': [1e-4,1e-3,1e-2,1e-2,1,10],
                    'svc__C':[1,50,100,200,300,500]}
clf = GridSearchCV(pipeline, hyperparameters,cv=8)
clf.fit(X_train, y_train)
end_time=time.time()

# make prediction on test set
y_pred = clf.predict(X_test)

# print performance of the model
acc=metrics.accuracy_score(y_pred,y_test)*100
print("Test accuracy is: %0.2f" % acc +"%.")
print("Confusion matrix is:")
print(metrics.confusion_matrix(y_pred, y_test, labels=[3,4,5,6,7,8,9]))
print("Best parameters are:")
print(clf.best_params_)
print("Running time is %0.2f seconds." % (end_time - start_time))
