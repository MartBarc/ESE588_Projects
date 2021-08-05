import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse
import scipy.stats

# Generate data using multivariate normal distribution (n_components=2)
N_1 = 100
N_2 = 20  # N_2 = 20 for second Dataset
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

# train GMM with EM
gmm = GaussianMixture(n_components=2, covariance_type='full').fit(x_train)

# clustering with K-means
k_means = KMeans(n_clusters=2, random_state=0).fit(x_train)

# plot for GMM AND EM
def draw_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()

    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))

ax = plt.gca()
labels = gmm.predict(x_train)
w_factor = 0.2 / gmm.weights_.max()
for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
    draw_ellipse(pos, covar, alpha=w * w_factor)

# Calculating the centroid of GMM
centers = np.empty(shape=(gmm.n_components, x_train.shape[1]))
for i in range(gmm.n_components):
    density = scipy.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(x_train)
    centers[i, :] = x_train[np.argmax(density)]
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, zorder=10)

print(centers)

# plot for K_means
plt.scatter(x_train[:, 0], x_train[:, 1], c=k_means.predict(x_train), s=50, marker='o', cmap='Spectral', linewidths=1)
plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], c='black', marker='x', s=200, linewidths=2, zorder=10)

# Centroid of K_means
print(k_means.cluster_centers_)

plt.title("GMM(O) vs K_means(X)")
plt.show()
