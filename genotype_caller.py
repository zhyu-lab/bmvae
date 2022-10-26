import numpy as np


class GenotypeCaller:
    def __init__(self, label, data):
        """
        self.T: Number of iterations to update genotype matrix, false positive rate and false positive rate
        self.data: Single cell mutation data
        self.label: inferred cell labels
        self.K：Number of clusters
        self.genotype_post_prob：Posterior probs of genotypes
        self.alpha: False positive rate
        self.beta: False negative rate
        self.min_alpha,self.min_beta: lower limit for false positive rate and false negative rate
        self.limit: a parameter to control the number of iterations for training
        """

        self.T = 500
        self.data = data
        self.label = label
        self.K = len(np.unique(label))

        self.genotype_post_prob = np.zeros([self.K, data.shape[1]], dtype='float32')
        self.genotypes = np.zeros([self.K, data.shape[1]], dtype='int32')

        self.alpha = 0.01
        self.beta = 0.01
        self.min_alpha = 1.0e-10
        self.min_beta = 1.0e-10

        self.limit = 1.0e-6

    def estimate_genotypes(self):
        post_prob_pre = np.zeros([self.K, self.data.shape[1]])
        for t in range(self.T):
            self.update_post_prob()
            self.update_alpha_beta()

            """
            training terminates if the mean absolute distance between the posteriors at two consecutive iterations
            is less than a predefined threshold
            """
            if t > 0:
                dist = 0
                for k in range(self.K):
                    dist += np.linalg.norm(post_prob_pre[k] - self.genotype_post_prob[k], ord=1)
                dist /= self.K * self.data.shape[1]
                if dist < self.limit:
                    break
                else:
                    post_prob_pre = self.genotype_post_prob
        self.genotypes = self.genotype_post_prob > 0.5
        return self.genotypes, self.alpha, self.beta

    def update_post_prob(self):
        """
        Update posterior probs of genotypes
        """
        # N, M = self.data.shape
        for k in range(self.K):
            tv_k = self.label == k
            d = self.data[tv_k, ]
            tv = np.logical_or(d == 0, d == 1)

            tmp1 = np.ones(d.shape, dtype='float128')
            tmp2 = np.ones(d.shape, dtype='float128')

            tmp1[tv] = np.power(1 - self.beta, d[tv]) * np.power(self.beta, 1 - d[tv])
            tmp2[tv] = np.power(1 - self.alpha, 1 - d[tv]) * np.power(self.alpha, d[tv])

            tmp3 = np.prod(tmp1, axis=0, dtype='float128')
            tmp4 = np.prod(tmp2, axis=0, dtype='float128')
            self.genotype_post_prob[k, ] = tmp3 / (tmp3 + tmp4)
            tv = np.random.random(d.shape[1]) <= self.genotype_post_prob[k, ]
            self.genotypes[k, tv] = 1
            self.genotypes[k, np.logical_not(tv)] = 0

    def update_alpha_beta(self):
        """
        Update false negative rate and false positive rate
        """
        g = self.genotypes[self.label, ]
        tv = np.logical_or(self.data == 0, self.data == 1)
        tmp1 = np.sum((1-g[tv])*self.data[tv])
        tmp2 = np.sum(1 - g[tv])
        tmp3 = np.sum(g[tv] * (1-self.data[tv]))
        tmp4 = np.sum(g[tv])
        tmp = tmp1 / tmp2
        if not np.isnan(tmp):
            self.alpha = tmp
        self.alpha = np.min([np.max([self.alpha, self.min_alpha]), 0.99])
        tmp = tmp3 / tmp4
        if not np.isnan(tmp):
            self.beta = tmp
        self.beta = np.min([np.max([self.beta, self.min_beta]), 0.99])

    def get_obs_prob(self, gt, observed):
        if observed == 0:
            return 1-self.alpha if gt == 0 else self.beta
        elif observed == 1:
            return self.alpha if gt == 0 else 1-self.beta
        else:
            return 1



