import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


class ConstrainedClustering:
    def __init__(self, eps=0.4, min_samples=4):
        self.eps = eps
        self.min_samples = min_samples
        self.neighbors_model = NearestNeighbors(radius=self.eps, metric='precomputed', n_jobs=-1)

    def fit(self, X, must_link=None, cannot_link=None):
        self.neighbors_model.fit(X)
        neighborhoods = self.neighbors_model.radius_neighbors(X, return_distance=False)
        labels = np.full(X.shape[0], -1, dtype=np.intp)
        cid = 0
        for point in tqdm(range(X.shape[0])):
            if labels[point] == -1 and \
                    self.expand_cluster(labels, cid, point, neighborhoods, must_link, cannot_link):
                cid += 1
        return labels

    def expand_cluster(self, labels, cid, point, neighborhoods, ml, cl):
        seeds = []
        neighb = neighborhoods[point]

        if len(neighb) < self.min_samples:
            return False

        if ml[point]:
            for p in ml[point]:
                if labels[p] == -1:
                    labels[p] = cid
                    seeds.append(p)
        else:
            labels[point] = cid
            seeds.append(point)

        while seeds:
            seed = seeds[0]
            if ml[seed]:
                for p in ml[seed]:
                    if labels[p] == -1:
                        labels[p] = cid
                        seeds.append(p)
            neighb = neighborhoods[seed]
            if len(neighb) >= self.min_samples:
                for p in neighb:
                    if labels[p] == -1 and self.check_cl_constraints(cl, p, labels, cid):
                        labels[p] = cid
                        seeds.append(p)
            del seeds[0]
        return True

    def check_cl_constraints(self, cl, point, labels, cluster_id):
        indices = np.where(labels == cluster_id)[0]
        for i in indices:
            if i in cl[point]:
                return False
        return True
