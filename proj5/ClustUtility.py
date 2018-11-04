import numpy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

class ClustUtility:

    def __init__(self):
        pass

    # Elbow method to pick up 'k'
    def Estimate_K_elbow(self, X, data_name, kmax='auto'):
        if kmax == 'auto':
            kmax = round(X.shape[0]/(X.shape[0]*0.05))
        lst_sse = 0
        optimal_k = None
        sse_iter = []
        for k in range(1, kmax):
            kmobj = KMeans(n_clusters=k, max_iter=100, random_state=1, n_jobs=4)
            kmobj.fit(X)
            un_clust = numpy.unique(kmobj.labels_)
            sse_iter.append(self.ComputeSSE(kmobj.labels_, kmobj.cluster_centers_, X)/len(un_clust))
            if optimal_k is None:
                if k == 1:
                    lst_sse = sse_iter[0]
                if k >= 2:
                    cost = sse_iter[k-1] + len(un_clust)*numpy.linalg.norm(un_clust, ord=1)
                    if cost > lst_sse:
                        optimal_k = k-2
                    else:
                        lst_sse = cost

        fig = plt.figure()
        plt.plot(range(1, kmax), sse_iter, 'bx-')
        plt.xlabel('K')
        plt.ylabel('MSE')
        plt.title('The Elbow method to estimate K')
        plt.axvline(x=optimal_k, color='red', linestyle='--')

        fig.savefig(data_name + "_elbow_determination_k.pdf", bbox_inches='tight')

        return optimal_k

    # Given, the original X, clustered X, and centers, compute SSE
    def ComputeSSE(self, cluster, centers, X):

        un_clust = numpy.unique(cluster)
        cl_count = 0
        error_ov = []
        for cl in un_clust:
            sub_X = X[sum(numpy.where(cluster == cl)), :]
            error_clust = []
            for i in range(0, sub_X.shape[0]):
                error = numpy.subtract(sub_X[i,:], centers[cl_count])**2
                error = sum(error)
                error_clust.append(error)
            error_ov.append(sum(error_clust))
            cl_count = cl_count + 1
        return numpy.sum(error_ov)

    # Compute centers of cluster, given original X and clustered X
    def ComputeCenters(self, cluster, X):
        un_clust = numpy.unique(cluster)
        center = []
        for cl in un_clust:
            sub_X = X[sum(numpy.where(cluster == cl)), :]
            mn_pts = []
            for i in range(0, sub_X.shape[1]):
                mn_pts.append(numpy.mean(sub_X[:, i]))
            center.append(mn_pts)
        return center

    # Compute min_pts and eps for DBSCAN
    def Estimate_eps_n_min_pts(self, X, K, data_name):
        samples = X.shape[0]
        nn = NearestNeighbors(n_neighbors=(K + 1))
        neighbors = nn.fit(X)
        distances, indices = neighbors.kneighbors(X)
        distance_mtx = numpy.empty([K, samples])
        min_pts = []
        eps = []
        for i in range(0, K):
            dst = distances[:, (i+1)]
            dst.sort()
            dst = dst[::-1]
            lst_cost = numpy.inf
            for j in range(0, samples):
                cost = dst[j] + 0.05*(numpy.linalg.norm(range(0, j), ord=2))
                if lst_cost < cost:
                    min_pts.append(j-1)
                    eps.append(dst[j-1])
                    break
                lst_cost = cost
            distance_mtx[i] = dst

        min_pts = min_pts[numpy.argmin(eps)]
        eps = eps[numpy.argmin(eps)]

        fig = plt.figure()
        for i in range(0, K):
            plt.plot(distance_mtx[i])
        plt.ylabel('distance')
        plt.xlabel('points')
        plt.title('Nearest neighbor method to estimate eps and min_pts')
        plt.axvline(x=min_pts, color='red', linestyle='--')
        plt.axhline(y=eps, color='red', linestyle='--')

        fig.savefig(data_name + "_nn_determination_minpts_eps.pdf", bbox_inches='tight')
        return min_pts, eps
