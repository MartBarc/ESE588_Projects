import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import scipy.stats
import copy

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
col1_13_ALL_names = ['CRIM', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']
col1_13_ALL = [col01_CRIM, col05_NOX, col06_RM, col07_AGE, col08_DIS, col10_TAX, col11_PTRATIO, col12_B, col13_LSTAT]
col_blank = [0, 0, 0, 0, 0, 0, 0, 0, 0]

# col1_13_ALL_names = ['CRIM', 'NOX']
# col1_13_ALL = [col01_CRIM, col05_NOX]
# col_blank = [0, 0]

# TARGET Y: Median value of the home
col14_MEDV = pd.DataFrame(data['MEDV'])

# N_samples = number of sample / rows
N_samples = col14_MEDV.size
# P_var_all = number of all features / cols
P_var_all = len(col1_13_ALL)

# [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1]
P_user_sel_array = [col_blank]


Best_Settings_R2 = col_blank
Best_Settings_R2_val = 0

Best_Settings_FTest = col_blank
Best_Settings_FTest_val = 0

Best_for_Both = col_blank
Best_for_Both_val = [0, 0]

ite = 1
while True:
    showPlot = False
    print_console = False

    P_user_last = copy.deepcopy(P_user_sel_array[ite - 1])
    if all(x == 1 for x in P_user_last):
        break
    for Bit_index in range(0, len(P_user_last)):
        carry_bit = False
        if P_user_last[Bit_index] == 0:
            P_user_last[Bit_index] = 1
            break
        else:
            carry_bit = True
            P_user_last[Bit_index] = 0
    P_user_sel_array.append(P_user_last)
    P_user_sel = P_user_last

    ite += 1

    P_user_sel_names = []

    P_var = 0
    for i in range(0, P_var_all):
        if P_user_sel[i] == 1:
            P_user_sel_names.append(col1_13_ALL_names[i])

            P_var += 1
    print(ite, P_user_sel_names)

    X_list = []
    for i in range(0, N_samples):
        X_row = []
        for j in range(0, P_var_all):
            if P_user_sel[j] == 1:
                X_row.append(col1_13_ALL[j].iloc[i][col1_13_ALL_names[j]])
        X_list.append(X_row)

    X = pd.DataFrame(X_list, columns=P_user_sel_names)
    Y = col14_MEDV

    # evaluating different models
    # using sklearn linear model
    model = linear_model.LinearRegression()
    model.fit(X, Y)

    B0_hat = model.intercept_[0]
    BN_hat = model.coef_[0]

    # print(B0_hat)
    # print(BN_hat)

    # y mean
    y_mean = [Y['MEDV'].mean() for i in range(N_samples)]

    # y_i
    y_i = data['MEDV']

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
        print('TEST1: Coefficient of determination (R2):', R2)
    if R2 > Best_Settings_R2_val:
        Best_Settings_R2 = copy.deepcopy(P_user_sel)
        Best_Settings_R2_val = copy.deepcopy(R2)

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
    if (F - Fa) > Best_Settings_FTest_val:
        Best_Settings_FTest = copy.deepcopy(P_user_sel)
        Best_Settings_FTest_val = copy.deepcopy((F - Fa))

    if R2 - Best_for_Both_val[0] >= 0 and (F - Fa) - Best_for_Both_val[1] >= 0:
        Best_for_Both = copy.deepcopy(P_user_sel)
        Best_for_Both_val = copy.deepcopy([R2, (F - Fa)])

    # plot residuals
    if showPlot:
        fig, axs = plt.subplots(P_var)
        fig.suptitle('TEST2: Residual Plot: Selected Features vs. Y')

        i = 0
        for col in X:
            axs[i].plot((X.iloc[:, i]), y_mean, color='green', linewidth=1)
            axs[i].scatter((X.iloc[:, i]), y_hat, color='red', s=10, marker="x")
            axs[i].scatter((X.iloc[:, i]), y_i, color='blue', s=10, marker="x")
            axs[i].set_title(X.columns[i] + ' vs MEDV')
            axs[i].set_ylabel('MEDV')
            axs[i].set_xlabel(X.columns[i])
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

    # input("Press Enter to continue...")
    if print_console:
        print('***********************')
        print(Best_Settings_R2)
        print(Best_Settings_R2_val)
        print(Best_Settings_FTest)
        print(Best_Settings_FTest_val)
        print(Best_for_Both)
        print(Best_for_Both_val)
        print('***********************')

print('***********************')
print(Best_Settings_R2)
print(Best_Settings_R2_val)
print(Best_Settings_FTest)
print(Best_Settings_FTest_val)
print(Best_for_Both)
print(Best_for_Both_val)
print('***********************')