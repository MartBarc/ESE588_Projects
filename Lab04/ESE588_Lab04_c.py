import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate data using multivariate normal distribution (n_components=2)
N_1 = 100
N_2 = 100 # N_2 = 20 for second Dataset
mu_1 = np.array([-1.0, -1.5])
mu_2 = np.array([1.0, 1.5])
sigma_1 = np.array([[1.0, 0.2],
                    [0.2,  1.0]])
sigma_2 = np.array([[2.0, 0.1],
                    [0.1, 2.0]])

np.random.seed(0)  # make sure random generations are consistent
data_1 = np.random.multivariate_normal(mean=mu_1, cov=sigma_1, size=N_1)
data_2 = np.random.multivariate_normal(mean=mu_2, cov=sigma_2, size=N_2)
x_train = np.vstack([data_1, data_2])

np.random.seed(0)  # make sure random generations are consistent
data_1 = np.random.multivariate_normal(mean=mu_1, cov=sigma_1, size=N_1)
data_2 = np.random.multivariate_normal(mean=mu_2, cov=sigma_2, size=N_2)
x_train = np.vstack([data_1, data_2])

# clustering with K-means
k_means = KMeans(n_clusters=2, random_state=0).fit(x_train)

print(k_means.labels_)
y_k_means = k_means.predict(x_train)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_k_means, s=25, cmap='Spectral')
centers = k_means.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.7);

plt.show()
