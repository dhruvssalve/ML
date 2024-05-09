import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import imblearn
import seaborn as sns
from sklearn import metrics
from imblearn.under_sampling import RandomUnderSampler
url="https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
data = pd.read_csv('data_banknote_authentication.txt',header=None)
data.columns = ['var','skew','curt','entr','auth']
print(data.head())
data.info()

x = data.drop('auth',axis=1)
y=data['auth']
target_count=data.auth.value_counts()
rus = RandomUnderSampler(random_state=42,replacement=True)
x_rus,y_rus = rus.fit_resample(x,y)
x_train,x_test,y_train,y_test = train_test_split(x_rus,y_rus,test_size=0.3,random_state=42)
scalar = StandardScaler()
scalar.fit(x_train)
x_train = scalar.transform(x_train)
x_test = scalar.transform(x_test)
clf = LogisticRegression(solver='lbfgs',random_state=42,multi_class='auto')
clf.fit(x_train,y_train.values.ravel())
y_pred = np.array(clf.predict(x_test))
conf_mat = pd.DataFrame(confusion_matrix(y_test,y_pred),columns=["Pred.Negative","Pred.Positive"],index=['Act.Negative','Act.Positive'])
tn,fp,fn,tp = confusion_matrix(y_test,y_pred).ravel()
accuracy = round((tn+tp)/(tn+fp+fn+tp),4)
print(conf_mat)
print(f'\n Accuracy = {round(100*accuracy,2)}%')

cf_matrix = metrics.confusion_matrix(y_test,y_pred)
sns.heatmap(cf_matrix,annot=True,fmt='')
from sklearn.metrics import ConfusionMatrixDisplay
cf_matrix = metrics.confusion_matrix(y_test,y_pred)
ConfusionMatrixDisplay(cf_matrix).plot()
