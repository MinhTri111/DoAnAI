import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X = []
Y = []
list_link = os.listdir('data/')
# print("list_link = ",list_link)

for i in list_link:
    link = 'data/{}'.format(i)
    link_img = os.listdir(link)
    for j in link_img:
        img = cv2.imread('data/{}/{}'.format(i,j), 0)
        img = cv2.resize(img,(30,60))
        img = img.reshape(1, -1)
        X.append(img[0].tolist())
        Y.append(int(i))

X = np.array(X,dtype=np.float32)
Y = np.array(Y)
sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2)
for train_index, test_index in sss.split(X, Y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

y_train = y_train.reshape(-1, 1)
# print("y_train =",y_train)

svm_model = cv2.ml.SVM_create()
svm_model.setType(cv2.ml.SVM_C_SVC)
svm_model.setKernel(cv2.ml.SVM_INTER)
svm_model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-10))
#opencv chi support tham so row_sample
svm_model.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

svm_model.save("model_svm2.xml")

a = svm_model.predict(X_test)
y_predicted = []
for i in a[1]:
    y_predicted.append(i[0])
y_predicted = np.array(y_predicted)
target = [chr(x) for x in y_predicted]
target = list(set(target))
target.sort()

#Su chinh xac
print('accuracy  = ',accuracy_score(y_test,y_predicted))
#Do chinh xac
print('precision = ', precision_score(y_test,y_predicted,average= 'weighted'))
print('recall    = ', recall_score(y_test,y_predicted,average= 'weighted'))
print("f1        =", f1_score(y_test,y_predicted, average= 'weighted'))
print(classification_report(y_test, y_predicted, target_names=target))
cm = confusion_matrix(y_test, y_predicted)
cm = pd.DataFrame(data = cm[0:,0:], index = target, columns = target)

plt.figure(figsize = (20,14))
sn.heatmap(cm, annot=True)
plt.xlabel("Predict")
plt.ylabel("Truth")
plt.show()