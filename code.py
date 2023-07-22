import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as skm
import scipy.stats as sps

df = pd.read_csv('SteelPlateFaults-2class.csv')
#spliting the data into test-train
[xtrain, xtest, xlabeltrain, xlabeltest] = train_test_split(df, df.Class, test_size=0.3, random_state=42, shuffle=True)
class2 = xtest.groupby('Class')
zero = class2.get_group(0)
one = class2.get_group(1)

#using KNN method and find confusion matrix and classification accuracy
for i in range(1,6,2):
    kn = KNeighborsClassifier(n_neighbors=i)
    kn.fit(xtrain.iloc[:,:27],xtrain.Class)
    y_pred = kn.predict(xtest.iloc[:,:27])
    y_true = xlabeltest.tolist()
    print("KNN")
    print(skm.confusion_matrix(y_true, y_pred))
    print(skm.accuracy_score(y_true, y_pred))
    
xtrain.to_csv('SteelPlateFaults-train.csv', index=False)
xtest.to_csv('SteelPlateFaults-test.csv', index=False)

#min-max normalization
xtrain1 = pd.read_csv("SteelPlateFaults-train.csv")
xtest1 = pd.read_csv("SteelPlateFaults-test.csv")
xtrain1.drop('Class',axis=1,inplace=True)
xtest1.drop('Class',axis=1,inplace=True)
for i in xtrain1.columns:
    mx = xtrain1[i].max()
    mn = xtrain1[i].min()
    xtrain1[i]=(xtrain1[i]-mn)/(mx-mn)
    xtest1[i]=(xtest1[i]-mn)/(mx-mn)
xtrain.to_csv('SteelPlateFaults-train-Normalised.csv', index=False)
xtest.to_csv('SteelPlateFaults-test-Normalised.csv', index=False)

#using KNN for finding confusion matrix and classification accuracy of normalised data
for i in range(1,6,2):
    kn = KNeighborsClassifier(n_neighbors=i)
    kn.fit(xtrain1,xlabeltrain)
    y_pred = kn.predict(xtest1)
    y_true = xlabeltest.tolist()
    print("KNN Normalised")
    print(skm.confusion_matrix(y_true, y_pred))
    print(skm.accuracy_score(y_true, y_pred))
    

y_true = xlabeltest.tolist()
xtrain2 = pd.read_csv("SteelPlateFaults-train.csv")
xtest2 = pd.read_csv("SteelPlateFaults-test.csv")
group0 = xtrain2.groupby('Class')
class0 = group0.get_group(0).iloc[:,:27]
class1 = group0.get_group(1).iloc[:,:27]
C0 = class0.shape[0]
C1 = class1.shape[0]
PC0 = C0/(C0 + C1)
PC1 = C1/(C0 + C1)
cov_mat0 = np.cov(class0.T)
cov_mat1 = np.cov(class1.T)
mean0 = np.array(class0.mean())
mean1 = np.array(class1.mean())
def mul_pdf(x,u,cov):
    d = x.shape[0]/2
    det = abs(np.linalg.det(cov))
    tr = np.transpose(x-u)
    inve = np.linalg.inv(cov)
    pdf = 1/(((2*np.pi)*d)*det*(1/2))
    pdf *= np.exp((-1/2)*np.matmul(np.matmul(tr,inve),(x-u)))
    return pdf

pred_bayes = []
xtest2.drop('Class',axis=1,inplace=True)
for i in range(xtest2.shape[0]):
    prob0 = mul_pdf(np.array(xtest2.iloc[i]),mean0,cov_mat0)*PC0
    prob1 = mul_pdf(np.array(xtest2.iloc[i]),mean1,cov_mat1)*PC1
    if prob0>prob1:
        pred_bayes.append(0)
    else:
        pred_bayes.append(1)

print('bayes')
print()
print(skm.confusion_matrix(y_true, pred_bayes))
print()
print(skm.accuracy_score(y_true, pred_bayes))
print()

#tabulating and comparing the best classsifier
from prettytable import PrettyTable
table = PrettyTable()
table.field_names = ['Classifier', 'Best Score']
table.add_rows(
[
    ["Naive Bayes", "0.6785"],
    ["kn-Normalised", "0.9672"],
    ["knn", "0.8958"],
])
print(table)