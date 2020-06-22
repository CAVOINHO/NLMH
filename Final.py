import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#Hàm xử lý giải thuật Bayes thơ ngây  sử dụng nghi thức Hold_out
def Hold_out(dirfile, n):
   dt = pd.read_csv("spambase.data", delimiter=",", header=None)
   thuoctinh = dt.iloc[:,0:57]
   nhan = dt.iloc[:, 57:58]
   K1_list = []
   Kq1_list = []
   train_list=[]
   avegrade = 0
   for i in range(0,n):
      X_train,X_test,y_train,y_test = train_test_split(thuoctinh, nhan , test_size=0.1, random_state=50*i)
      model = GaussianNB()
      model.fit(X_train, y_train.values.ravel())
      thucte = y_test
      dubao = model.predict(X_test)
      K1 = round(accuracy_score(y_test, dubao)*100, 2)
      K1_list.append(K1)
      avegrade += K1
      kq1 =  confusion_matrix(thucte, dubao)
      Kq1_list.append(kq1)
   return [len(dt), X_train, y_test, K1_list, kq1, avegrade/n]

#Hàm xử lý giải thuật Cây quyết định sử dụng nghi thức Hold_out
def DT(dirfile, n):
   dt = pd.read_csv("spambase.data", delimiter=",", header=None)
   thuoctinh = dt.iloc[:,0:57]
   nhan = dt.iloc[:, 57:58]
   tong = 0
   K2_list = []
   Kq2_list = []
   train_list = []
   for i in range(0,n):
      X_train,X_test,y_train,y_test = train_test_split(thuoctinh, nhan , test_size=0.1, random_state=50*i)
      clf_gini = DecisionTreeClassifier(criterion="gini", random_state=50, max_depth=4, min_samples_leaf=5)
      clf_gini.fit(X_train, y_train)
      y_pred = clf_gini.predict(X_test)
      K2 =  round(accuracy_score(y_test, y_pred)*100, 2)
      tong += K2
      K2_list.append(K2)
      kq2 =  confusion_matrix(y_test, y_pred)
      Kq2_list.append(kq2)
   return [len(dt), X_train, y_test, K2_list, kq2, tong/n]

#Hàm vẽ biểu đồ kết quả độ chính xác tổng thể của từng vòng lặp Bayes
def Chart_main(dirfile, n):
   a,b,c,K1_list,kq1,tongtbby = Hold_out(dirfile, n)
   x_values = [1,2,3,4,5,6,7,8,9,10]
   y_values = [82, 83, 84, 81, 83, 83, 82, 84, 83, 83]
   division_marks = K1_list
   plt.bar(x_values, division_marks,  color='darkcyan')

   #Ghi dữ liệu lên biểu đồ
   for i in range(len(K1_list)):
      plt.text(x = x_values[i]-0.25,
      y = y_values[i],
      s = division_marks[i],
      size = 10)
   plt.title("Sơ đồ so sánh")
   plt.xlabel("Kết quả của lần lặp")
   plt.ylabel("Độ chính xác (%)")
   plt.show()

#Hàm vẽ biểu đồ hình tròn chỉ số nhãn 1 và nhãn 0
def Chart_pie(dirfile, n):
   dt = pd.read_csv("spambase.data", delimiter=",", header=None)
   nhan = dt.iloc[:, 57:58]
   k = 0
   for i in range(len(nhan)):
      if(nhan.values[i] == 1):
         k += 1
   h = len(nhan) - k
   firms = ["Số lượng nhãn 1: 1813" , "Số lượng nhãn 0: 2788"]
   market_marks = [k, h]
   Explode = [0,0.06]
   plt.pie(market_marks, explode=Explode,labels=firms, shadow=True, startangle=35, autopct='%1.1f%%')
   plt.axis('equal')
   plt.show()

#Hàm vẽ biểu đồ so sánh độ chính xác của 2 giải thuật Bayes thơ ngây và Cây quyết định
def Chart_column(dirfile, n):
   a,b,c,K1_list,kq1,tongtbby = Hold_out(dirfile, n)
   _,_,_,K2_list,kq2,tongtbdt = DT(dirfile, n)
   x_values = [0,1,2]
   x = ["HOLD OUT BAYES", "HOLD OUT DT"]
   y_values = [5, 10]
   division_marks = [round(tongtbby,2), round(tongtbdt,2)]
   plt.bar(x, division_marks, color='darkcyan')

   #Ghi dữ liệu lên biểu đồ
   for i in range(len(division_marks)):
      plt.text(x = x_values[i] - 0.15,
      y = y_values[i]+80,
      s = division_marks[i],
      size = 10)
   plt.title("Sơ đồ so sánh")
   plt.xlabel("Phương thức sử dụng")
   plt.ylabel("Độ chính xác (%)")
   plt.show()

   #In các thông số và kết quả
   print('Số lượng phần tử tập dữ liệu: ', a)
   print('Số lượng phần tử tập huấn luyện: ', len(b))
   print('Số lượng phần tử tập kiểm tra: ', len(c))
   print("")
   print('ĐỘ CHÍNH XÁC CỦA GIẢI THUẬT')
   print('Hold out: ', round(tongtbby,2), "%")
   print('Độ chính xác của từng phân lớp: \n', '[',np.unique(c)[0],'  ',np.unique(c)[1],']')
   print(kq1)
   print("")
   print('Số lượng phần tử tập dữ liệu: ', a)
   print('Số lượng phần tử tập huấn luyện: ', len(b))
   print('Số lượng phần tử tập kiểm tra: ', len(c))
   print("")
   print('___________________________')
   print('ĐỘ CHÍNH XÁC CỦA TỪNG GIẢI THUẬT')
   print('Bayes thơ ngây: ', round(tongtbby,2), "%")
   print('Decision Tree: ', round(tongtbdt,2), "%")

def main():
   dt = pd.read_csv("spambase.data", delimiter=",", header=None)
   # print(dt)
   dirfile = 'D:/BC/spambase.data' #Vị trí file dữ liệu, phải đúng vị trí
   Chart_main(dirfile, 10)
   Chart_pie(dirfile, 10)
   Chart_column(dirfile, 10)
main()
