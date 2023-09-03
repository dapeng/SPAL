import torch
import numpy as np


class ActiveLearning:
    def __init__(self, num_instances, budget):
        self.num_instances = num_instances
        self.budget = int(budget * self.num_instances)
        self.ml_graph = dict()
        self.cl_graph = dict()
        for i in range(self.num_instances):
            self.ml_graph[i] = set()
            self.cl_graph[i] = set()

    @staticmethod
    def _add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    def _add_ml(self, i, j):
        self._add_both(self.ml_graph, i, j)
        visited = [False] * self.num_instances
        if self.ml_graph[i]:
            component = []
            self.dfs(i, self.ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2:
                        self.ml_graph[x1].add(x2)
                        for x3 in self.cl_graph[x2]:
                            self._add_both(self.cl_graph, x1, x3)

    def _add_cl(self, i, j):
        self._add_both(i, j)
        for x in self.ml_graph[i]:
            self._add_both(x, j)
        for y in self.ml_graph[j]:
            self._add_both(i, y)
        for x in self.ml_graph[i]:
            for y in self.ml_graph[j]:
                self._add_both(x, y)

    def dfs(self, i, graph, visited, component):
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                self.dfs(j, graph, visited, component)
        component.append(i)

    def get_masks(self):
        masks = torch.zeros(self.num_instances, self.num_instances, dtype=torch.int)
        for i in range(self.num_instances):
            masks[i, list(self.ml_graph[i])] = 1
            masks[i, list(self.cl_graph[i])] = -1
        return masks

    def dus(self, pseudo_labels, gt_labels, dist_mat):
        """
        Dual Uncertainty Selection
        :param pseudo_labels:
        :param gt_labels:
        :param dist_mat:
        :return:
        """
        if self.budget == 0:
            return

        cluster_ids = set(pseudo_labels)
        if -1 in cluster_ids:
            cluster_ids.remove(-1)
        for id in cluster_ids:
            sp = []
            intra_idx = np.where(pseudo_labels == id)[0]
            inter_idx = np.where(pseudo_labels != id)[0]

            # uncertain postive pairs
            intra_dist_mat = dist_mat[np.ix_(intra_idx, intra_idx)]
            pos = np.unravel_index(intra_dist_mat.argmax(), intra_dist_mat.shape)
            sp.append((intra_idx[pos[0]], intra_idx[pos[1]]))

            # uncertain negative pairs
            inter_dist_mat = dist_mat[np.ix_(intra_idx, inter_idx)]
            neg = np.unravel_index(inter_dist_mat.argmin(), inter_dist_mat.shape)
            sp.append((intra_idx[neg[0]], inter_idx[neg[1]]))

            for idx, (u, v) in enumerate(sp):
                if self.budget == 0 or u == v or u in self.ml_graph[v] or u in self.cl_graph[v]:
                    continue
                if gt_labels[u] == gt_labels[v]:
                    self._add_ml(u, v)
                else:
                    self._add_cl(u, v)
                self.budget -= 1
