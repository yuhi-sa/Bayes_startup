# -*- coding: utf-8 -*-

# %matplotlib inline
import numpy as np
from scipy.stats import norm
import math
import matplotlib.pyplot as plt

#基底関数
def phi(x):
    return [x, x**2, x**3 ,x**4, x**5 ,x**6 ,x**7, x**8, x**9]

dataset = np.load('dataset.npz')

#ハイパーパラメータ
alpha = 10
beta =10

#トレーニングデータ
x_train = dataset['x_train']
y_train = dataset['y_train']

#テストデータ
x_test = dataset['x_test']
y_test = dataset['y_test']

#ベイズ推定
PHI = np.array([phi(x) for x in x_train])
w = np.dot(np.linalg.inv((alpha/beta)+np.dot(PHI.T,PHI)),np.dot(PHI.T,y_train))
#y = np.dot(PHI,np.expand_dims(u,0).T)


PHI = np.array([phi(x) for x in x_test])
u = np.dot(np.linalg.inv((alpha/beta)+np.dot(PHI.T,PHI)),np.dot(PHI.T,y_test))
sigma = np.linalg.inv(alpha + beta*np.dot(PHI.T,PHI))
sigma2 = 1/beta - np.dot(np.dot(PHI,sigma),PHI.T)

y_predict =np.dot(PHI,w.T)

s = 1.96*np.mean(np.sqrt(sigma))

predict_upper = y_predict + s
predict_lower = y_predict - s


#決定係数
y_test_mean=np.mean(y_test)
a=y_test-y_test_mean
b=y_test-y_predict

R2=1- np.sum(b*b)/np.sum(a*a)
print(R2)

#プロット
fig, ax = plt.subplots(1,2,figsize=(12,4))
ax[0].fill_between(x_test,predict_upper,predict_lower,label='confidence interval',facecolor='y')
ax[1].fill_between(x_test,predict_upper,predict_lower,label='confidence interval',facecolor='y')

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
