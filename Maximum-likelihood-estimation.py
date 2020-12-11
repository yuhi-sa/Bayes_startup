# -*- coding: utf-8 -*-

# %matplotlib inline
import numpy as np
import math
import matplotlib.pyplot as plt

#基底関数
def phi(x):
    return [x, x**2, x**3 ,x**4, x**5 ,x**6 ,x**7, x**8, x**9]

dataset = np.load('dataset.npz')

#トレーニングデータ
x_train = dataset['x_train']
y_train = dataset['y_train']

#テストデータ
x_test = dataset['x_test']
y_test = dataset['y_test']

#最尤推定
PHI = np.array([phi(x) for x in x_train])
w = np.dot(np.linalg.inv(np.dot(PHI.T, PHI)), np.dot(PHI.T, y_train))

PHI = np.array([phi(x) for x in x_test])
e = (y_test-np.dot(PHI,w))
e2 = np.dot(e.T,e)/y_test.size
e3 = (1/(2*math.pi*e2)**(y_test.size)) * (math.e**(-np.dot(y_test.T,y_test)/(2*e2)))

y_predict =np.dot(PHI,w.T)+e3

#決定係数
y_test_mean=np.mean(y_test)
a=y_test-y_test_mean
b=y_test-y_predict

R2=1- np.sum(b*b)/np.sum(a*a)
print(R2)

#プロット
fig, ax = plt.subplots(1,2,figsize=(12,4))
ax[0].scatter(x_train, y_train, label='train_data',color='b')
ax[0].plot(x_test,y_predict, label='predict_data',color='r')

ax[1].scatter(x_test,y_test, label='test_data',color='0.1')
ax[1].plot(x_test,y_predict, label='predict_data',color='r')

ax[0].set_title('train')
ax[1].set_title('test')

ax[0].set_xlim(-1.5,1.5)
ax[1].set_ylim(-0.5,1.5)

ax[1].set_xlim(-1.5,1.5)
ax[1].set_ylim(-0.5,1.5)

ax[0].legend(loc='upper right',fontsize=9)
ax[1].legend(loc='upper right',fontsize=9)

plt.show()
