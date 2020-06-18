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

thuoctinh = dt.iloc[:,0:57]
nhan = dt.iloc[:, 57:58]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(thuoctinh, nhan , test_size=0.1, random_state=50)

print("Số lượng phần tử trong tập huấn luyện: ", len(X_train))
print("Số lượng phần tử trong tập kiểm tra: ", len(X_test))

print("___________________________________________________________________")
print("")
print("HOLD ON BAYES")

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

model = GaussianNB()
model.fit(X_train, y_train.values.ravel())

thucte = y_test
dubao = model.predict(X_test)

from sklearn.metrics import accuracy_score
K1 = round(accuracy_score(y_test, dubao)*100, 2)
print("Kết quả của giải thuật Bayes thơ ngây nghi thức hold on: ",K1,"%")

from sklearn.metrics import confusion_matrix

kq1 =  confusion_matrix(thucte, dubao)

print("Độ chính xác từng lớp: \n",'[',np.unique(nhan)[0],'  ',np.unique(nhan)[1],']')
print(kq1)
print("___________________________________________________________________")
print("")
print("HOLD ON DECISION TREE")
from sklearn.tree import DecisionTreeClassifier
clf_gini = DecisionTreeClassifier(criterion="gini", random_state=50, max_depth=4, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)
from sklearn.metrics import accuracy_score
K2 =  round(accuracy_score(y_test, y_pred)*100, 2)
print("Kết quả của giải thuật Cây quyết định: ",K2,"%")

from sklearn.metrics import confusion_matrix

kq2 =  confusion_matrix(y_test, y_pred)

print("Độ chính xác từng lớp: \n",'[',np.unique(nhan)[0],'  ',np.unique(nhan)[1],']')
print(kq2)
print("___________________________________________________________________")
print("")
print("K_FOLD BAYES")
from sklearn.model_selection import KFold
kf= KFold(n_splits=10, shuffle=True, random_state=None)

X = dt.iloc[:,0:57]
Y = dt.iloc[:,57:58]
tong = 0
t = 0
for train_index, test_index in kf.split(X):
   X_train, X_test = X.iloc[train_index,], X.iloc[test_index,]
   y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
   from sklearn.naive_bayes import GaussianNB
   from sklearn.naive_bayes import MultinomialNB
   model = GaussianNB()
   model.fit(X_train, y_train.values.ravel())
   thucte = y_test
   dubao = model.predict(X_test)
   from sklearn.metrics import confusion_matrix
   kq3 = confusion_matrix(thucte, dubao)
   t+=1;
   print("Lần lặp thứ",t,":" )
   print("Độ chính xác từng lớp: \n",'[',np.unique(nhan)[0],'  ',np.unique(nhan)[1],']')
   print(kq3)
   print("")
   from sklearn.metrics import accuracy_score
   j = accuracy_score(thucte, dubao)*100
   tong = tong + j

K3 = round(tong/10,2)
print("Độ chính xác tổng thể của trung bình 10 lần lặp: ",K3,"%")

import matplotlib.pyplot as plt

divisions = ["HOLD ON BAYES", "HOLD ON DT", "K_FOLD BAYES"]
division_marks = [K1, K2, K3]
plt.bar(divisions, division_marks, color='pink')
plt.title("Sơ đồ so sánh")
plt.xlabel("Phương thức sử dụng")
plt.ylabel("Độ chính xác")
plt.show()

firms = ["Số lượng nhãn 1", "Số lượng nhãn 0"]
market_marks = [k, h]
Explode = [0,0.06]
plt.pie(market_marks, explode=Explode,labels=firms, shadow=True, startangle=35)
plt.axis('equal')
plt.legend(title="Thành phần nhãn")
plt.show()

# firms = ["X_train", "X_test", "y_train", "y_test"]
# market_marks = [len(X_train), len(X_test), len(y_train), len(y_test)]
# Explode = [0.1 ,0.1,0.1 ,0.1]
# plt.pie(market_marks, explode=Explode,labels=firms, shadow=True, startangle=35)
# plt.axis('equal')
# plt.legend(title="Thành phần tập phân chia")
# plt.show()
