import pandas as pd
import numpy as np

dt = pd.read_csv("spambase.data", delimiter=",", header=None)

print("Số lượng phần tử tập dữ liệu: ", len(dt))

k = 0
h =0
nhan = dt.iloc[:, 57:58]
for i in range(len(nhan)):
   if(nhan.values[i] == 1):
      k = k + 1
   else:
      h = h + 1

print("Số lương phần tử nhã 0: ", h)
print("Số lương phần tử nhã 1: ", k)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split( dt.iloc[:,0:57], dt.iloc[:, 57:58], test_size=0.1, random_state=50)

print("Số lượng phần tử trong tập huấn luyện: ", len(X_train))
print("Số lượng phần tử trong tập kiểm tra: ", len(X_test))

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

model = GaussianNB()
model.fit(X_train, y_train.values.ravel())

thucte = y_test
dubao = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Kết quả của giải thuật Bayes thơ ngây nghi thức hold on: ", round(accuracy_score(y_test, dubao)*100, 2))

from sklearn.metrics import confusion_matrix

kq =  confusion_matrix(thucte, dubao)

print("Độ chính xác từng lớp: \n", kq)
