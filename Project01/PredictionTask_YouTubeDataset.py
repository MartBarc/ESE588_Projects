import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import scipy.stats

data = pd.read_csv('USvideos.csv')

# TREND_DATE,TITLE,CHAN_TITLE,CATEGORY,PUBLISH_TIME,TAGS,VIEWS,LIKES,DISLIKES, COMMENTS

col01_TREND_DATE = pd.DataFrame(data['TREND_DATE'])
col02_TITLE = pd.DataFrame(data['TITLE'])
col03_CHAN_TITLE = pd.DataFrame(data['CHAN_TITLE'])
col04_CATEGORY = pd.DataFrame(data['CATEGORY'])
col05_PUBLISH_TIME = pd.DataFrame(data['PUBLISH_TIME'])
col06_TAGS = pd.DataFrame(data['TAGS'])
#col07_VIEWS = pd.DataFrame(data['VIEWS'])
col08_LIKES = pd.DataFrame(data['LIKES'])
col09_DISLIKES = pd.DataFrame(data['DISLIKES'])
col10_COMMENTS = pd.DataFrame(data['COMMENTS'])

# data[['TREND_DATE', 'TITLE', 'CHAN_TITLE', 'CATEGORY', 'PUBLISH_TIME', 'TAGS', 'VIEWS',
# 'LIKES', 'DISLIKES', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
col1_10_ALL_names = ['TREND_DATE', 'TITLE', 'CHAN_TITLE', 'CATEGORY', 'PUBLISH_TIME', 'TAGS',
                     'LIKES', 'DISLIKES', 'COMMENTS'] #'VIEWS
col1_10_ALL = [col01_TREND_DATE, col02_TITLE, col03_CHAN_TITLE, col04_CATEGORY, col05_PUBLISH_TIME,
               col06_TAGS, col08_LIKES, col09_DISLIKES, col10_COMMENTS]#col07_VIEWS

# TARGET Y: Number of views
col07_VIEWS = pd.DataFrame(data['VIEWS'])

# N_samples = number of sample / rows
N_samples = 2300
# P_var_all = number of all features / cols
P_var_all = len(col1_10_ALL)

showPlot = True
print_console = True

P_user_sel = [0, 0, 0, 1, 0, 0, 1, 1, 1]

P_user_sel_names = []
P_var = 0
for i in range(0, P_var_all):
    if P_user_sel[i] == 1:
        P_user_sel_names.append(col1_10_ALL_names[i])
        P_var += 1
print(P_user_sel_names)

X_list = []
for i in range(0, N_samples):
    X_row = []
    for j in range(0, P_var_all):
        if P_user_sel[j] == 1:
            X_row.append(col1_10_ALL[j].iloc[i][col1_10_ALL_names[j]])
    X_list.append(X_row)

print(N_samples)
print(len(col07_VIEWS.index))

col07_VIEWS.drop(col07_VIEWS.index[N_samples:len(col07_VIEWS.index)], 0, inplace=True)

X = pd.DataFrame(X_list, columns=P_user_sel_names)
Y = col07_VIEWS

# evaluating different models
# using sklearn linear model
model = linear_model.LinearRegression()
model.fit(X, Y)

B0_hat = model.intercept_[0]
BN_hat = model.coef_[0]

# print(B0_hat)
# print(BN_hat)

# y mean
y_mean = [Y['VIEWS'].mean() for i in range(N_samples)]

# y_i
y_i = data['VIEWS']

y_i.drop(y_i.index[N_samples:len(y_i.index)], 0, inplace=True)

# y_hats
y_hat = [0 for i in range(N_samples)]
for i in range(0, N_samples):
    BN_tot = 0
    for p in range(0, P_var):
        BN_tot += BN_hat[p] * list(X.iloc[:, p])[i]
    y_hat[i] = B0_hat + BN_tot

# Get Syy
Syy = 0
for i in range(0, N_samples):
    Syy += (y_i[i] - y_mean[i]) ** 2

# Get SSE for each P
SSE = 0
for i in range(0, N_samples):
    SSE += (y_i[i] - y_hat[i]) ** 2

# SSE vs Syy
# if SSE < Syy:
#     print('Is SSE SMALLER than Syy?: Yes')
# else:
#     print('Is SSE SMALLER than Syy?: No')

# Print coeff
R2 = 1 - (SSE/Syy)
if print_console:
    print('TEST1: Coefficient of (R2):', R2)

# plot residuals old
# fig = plt.figure(0)
x_i = [i + 1 for i in range(0, N_samples)]
# fig.suptitle('n vs y_i', fontsize=10)
# plt.plot(x_i, y_mean, color='green', linewidth=1)
# plt.scatter(x_i, y_hat, color='red', linewidth=1)
# plt.scatter(x_i, y_i, color='blue', linewidth=1)
# for i in range(0, N_samples):
#    point1 = [x_i, y_hat[i]]
#    point2 = [x_i, y_i[i] + y_hat[i]]
#    plt.plot([point1[0], point2[0]], [point1[1], point2[1]], color='blue', linewidth=0.5)

# significance level
alpha = 0.05
degreesFreedom = N_samples - (P_var + 1)
F = ((Syy - SSE)/P_var) / (SSE/degreesFreedom)
Fa = scipy.stats.f.ppf(q=1-alpha, dfn=P_var, dfd=degreesFreedom)
if print_console:
    print('TEST3: F-Test:', F, '>', Fa)

# plot residuals
if showPlot:
    i = 0
    for col in X:
        fig = plt.figure(i)
        fig.suptitle(X.columns[i] + ' vs VIEWS', fontsize=12)
        plt.xlabel(X.columns[i], fontsize=10)
        plt.ylabel('VIEWS', fontsize=10)
        plt.plot((X.iloc[:, i]), y_mean, color='green', linewidth=1)
        plt.scatter((X.iloc[:, i]), y_hat, color='red', s=10, marker="x")
        plt.scatter((X.iloc[:, i]), y_i, color='blue', s=10, marker="o")
        i += 1
        plt.show()

# i = 0
# for col in X:
#    fig = plt.figure(i)
#    fig.suptitle(X.columns[i] + ' vs MEDV', fontsize=10)
#    plt.plot(list(X.iloc[:, i]), Y_mean_line, color='green', linewidth=1)
#    plt.scatter(list(X.iloc[:, i]), y_hat[i], color='red', linewidth=1)
#    plt.scatter(list(X.iloc[:, i]), Y_col, color='blue', linewidth=1)
#    i += 1
#    plt.show()
