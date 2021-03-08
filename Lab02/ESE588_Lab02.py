# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

data = pd.read_csv('BostonHousingData.csv')
# print(data.keys())

# CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,MEDV

col01_CRIM = pd.DataFrame(data['CRIM'])
col02_ZN = pd.DataFrame(data['ZN'])
col03_INDUS = pd.DataFrame(data['INDUS'])
col04_CHAS = pd.DataFrame(data['CHAS'])
col05_NOX = pd.DataFrame(data['NOX'])
col06_RM = pd.DataFrame(data['RM'])
col07_AGE = pd.DataFrame(data['AGE'])
col08_DIS = pd.DataFrame(data['DIS'])
col09_RAD = pd.DataFrame(data['RAD'])
col10_TAX = pd.DataFrame(data['TAX'])
col11_PTRATIO = pd.DataFrame(data['PTRATIO'])
col12_B = pd.DataFrame(data['B'])
col13_LSTAT = pd.DataFrame(data['LSTAT'])

# data[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
col1_13_ALL = [col01_CRIM, col02_ZN, col03_INDUS, col04_CHAS, col05_NOX, col06_RM, col07_AGE, col08_DIS, col09_RAD, col10_TAX, col11_PTRATIO, col12_B, col13_LSTAT]

# Feature columns 1, 5, 6, 7, 8, 11, 13
col1_5_6_7_8_13 = [col01_CRIM, col05_NOX, col06_RM, col07_AGE, col08_DIS, col13_LSTAT]

# pd.concat(col1_13_ALL)

# TARGET Y: Median value of the home
col14_MEDV = pd.DataFrame(data['MEDV'])
# combine data frames

# N_samples = number of sample / rows
N_samples = col14_MEDV.size
# P_var = number of selected features / cols
P_var = len(col1_5_6_7_8_13)
# P_var_all = number of all features / cols
P_var_all = len(col1_13_ALL)

# Create empty array of 0s for storing the feature estimated values
B_hat_Array = [0] * P_var

# linear regression model
model_man = linear_model.LinearRegression()

# get coefficients per col
# for iter in range(0, P_var - 1):
#     model_man.fit(col1_5_6_7_8_13[iter], col14_MEDV)
#     B_hat_Array[iter] = model_man.coef_

# using sklearn linear model
model_sk = linear_model.LinearRegression()

X = data[['CRIM', 'NOX', 'RM', 'AGE', 'DIS', 'LSTAT']]
Y = data[['MEDV']]

# using sklearn
model_sk.fit(X, Y)

B_hat_Array_model = model_sk.coef_
B0 = model_sk.intercept_

# get predictions
xnew = [22, 0.7, 7.5, 70, 9, 30]
y = 8.75

# plot
fig, axs = plt.subplots(6)
fig.suptitle('Features vs MEDV')

axs[0].scatter(col01_CRIM, col14_MEDV, s=10, marker="o")
axs[0].set_ylabel('MEDV')
axs[0].set_title('CRIM')
axs[0].plot(xnew[0], y, 'ro')

axs[1].scatter(col05_NOX, col14_MEDV, s=10, marker="o")
axs[1].set_title('NOX')
axs[1].plot(xnew[1], y, 'ro')

axs[2].scatter(col06_RM, col14_MEDV, s=10, marker="o")
axs[2].set_title('RM')
axs[2].plot(xnew[2], y, 'ro')

axs[3].scatter(col07_AGE, col14_MEDV, s=10, marker="o")
axs[3].set_title('AGE')
axs[3].plot(xnew[3], y, 'ro')

axs[4].scatter(col08_DIS, col14_MEDV, s=10, marker="o")
axs[4].set_title('DIS')
axs[4].plot(xnew[4], y, 'ro')

axs[5].scatter(col13_LSTAT, col14_MEDV, s=10, marker="o")
axs[5].set_title('LSTAT')
axs[5].plot(xnew[5], y, 'ro')

# data.plot(kind='scatter', x='MEDV', y='CRIM')
# plt.plot(col14_MEDV, col01_CRIM, color='blue', linestyle='None')
plt.show()
