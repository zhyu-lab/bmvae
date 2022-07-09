import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import operator


class G_Cluster():
    def __init__(self, data, Kmax):
        self.X = data
        self.x = self.X[:, 0]
        self.y = self.X[:, 1]
        self.z = self.X[:, 2]
        self.Kmax = Kmax


    def cluster(self):
        n_components_range = np.arange(1, self.Kmax+1)
        bic = []
        min_bic = []
        n_components = 1
        count = 0
        while n_components <= self.Kmax:
            gmm = GaussianMixture(n_components=n_components, covariance_type='full', n_init=50)
            gmm.fit(self.X)
            s = gmm.bic(self.X)
            bic.append(s)
            if n_components == 1 or s < min_bic:
                min_bic = s
                count = 0
            elif s >= min_bic:
                count += 1
            if count >= 10:
                break
            n_components += 1

        ind, val = min(enumerate(bic), key=operator.itemgetter(1))
        # fig = plt.figure(figsize=(6, 6))
        # plt.plot(np.arange(1, len(bic) + 1), bic)
        # plt.xlabel('Number of clusters')
        # plt.ylabel('BIC')
        # plt.show()

        gmm = GaussianMixture(n_components=ind + 1, covariance_type='tied')
        gmm.fit(self.X)
        probs = gmm.predict_proba(self.X)
        labels = gmm.predict(self.X)

        plt.figure("3D Scatter", facecolor="lightgray")
        ax3d = plt.gca(projection="3d")  
        plt.title('3D Scatter', fontsize=20)
        ax3d.set_xlabel('x', fontsize=14)
        ax3d.set_ylabel('y', fontsize=14)
        ax3d.set_zlabel('z', fontsize=14)
        plt.tick_params(labelsize=10)
        ax3d.scatter(self.x, self.y, self.z, s=20, c=[labels], cmap='viridis', marker="o")
        plt.show()
        return labels, n_components_range[ind]
