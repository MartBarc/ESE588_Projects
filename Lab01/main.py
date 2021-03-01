# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

data = pd.read_csv('fire_damage_data.csv')
data.plot(kind='scatter', x='Distance', y='Fire Damage')

distance = pd.DataFrame(data['Distance'])
damage = pd.DataFrame(data['Fire Damage'])

# calculations
n = len(distance)
x_distance = data['Distance']
y_damage = data['Fire Damage']

x_bar = 0.0
for i in x_distance:
    x_bar += i

y_bar = 0.0
for i in y_damage:
    y_bar += i

x_bar = x_bar / n
y_bar = y_bar / n

x_sum_squared = 0.0
for i in x_distance:
    x_sum_squared += i * i

xy_sum = 0.0
for i in range(0, n):
    xy_sum += x_distance[i] * y_damage[i]

S_xy = xy_sum - (n * x_bar * y_bar)
S_xx = x_sum_squared - (n * (x_bar * x_bar))

B1_hat = S_xy / S_xx
B0_hat = y_bar - B1_hat * x_bar

x_range = x_distance
y_hat = B0_hat + B1_hat * x_range

e_hat = [0] * n
for i in range(0, n):
    e_hat[i] = y_damage[i] - y_hat[i]

SEE = 0
for i in e_hat:
    SEE += i*i

noise_var = SEE/(n-2)
B1_hat_var = noise_var*noise_var/S_xx
B0_hat_var = noise_var*noise_var*(1/n + x_bar/S_xx)

for i in range(0, n):
    point1 = [x_distance[i], y_hat[i]]
    point2 = [x_distance[i], e_hat[i] + y_hat[i]]
    plt.plot([point1[0], point2[0]], [point1[1], point2[1]], color='blue')

# SLR line
# plt.plot(distance, SLR_model.predict(distance), color='red', linewidth=3)
plt.plot(x_distance, y_hat, color='blue', linewidth=1)
# plt.scatter(x_distance, e_hat, color='black')
# plt.scatter(future_distance, predict_damage, color='black')


# plt.plot(X, Y, color='blue', linewidth=4)
plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
