from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
import numpy
import time


class ClusteringAlgorithmPool:

    # common parameters
    random_state = None
    n_jobs = None
    dist_metric = None
    n_cluster = None
    runtime = None

    # K means
    kmeans_obj = None

    # Common parameter for Hierarchical clustering
    linkage_method = None

    # Hierarchical clust from scipy
    hier_scipy_obj = None

    # Agglomerative
    agglo_obj = None

    # DBSCAN
    eps_dbscan = None
    min_pts_dbscan = None
    dbscan_obj = None

    # Setting up variables and objects.
    def __init__(self, n_cluster, random_state, n_jobs, dist_metric, linkage_method, eps_dbscan, min_pts_dbscan):
        self.n_cluster = n_cluster
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.dist_metric = dist_metric
        self.runtime = numpy.zeros(4)

        # K means
        self.kmeans_obj = KMeans(n_clusters=self.n_cluster, max_iter=100, random_state=self.random_state,
                                 n_jobs=self.n_jobs)
        # Hierarchical clust
        self.linkage_method = linkage_method
        self.agglo_obj = AgglomerativeClustering(n_clusters= self.n_cluster, affinity=self.dist_metric,
                                                 linkage=self.linkage_method)

        # DBSCAN
        self.eps_dbscan = eps_dbscan
        self.min_pts_dbscan = min_pts_dbscan
        self.dbscan_obj =  DBSCAN(eps=self.eps_dbscan, min_samples=self.min_pts_dbscan, metric=self.dist_metric,
                                  algorithm ='kd_tree', n_jobs = self.n_jobs)

    def kmeans_fit(self, X):
        # Record fitting runtime
        start_time = time.time()
        print("Clustering X using K-means")
        self.kmeans_obj.fit(X)
        self.runtime[0] = time.time() - start_time
        print("Clustering ended in " + "{0:.2f}".format(round(self.runtime[0], 2)) + " seconds")

    def hierarchy_scipy_fit(self, X):
        # Record fitting runtime
        start_time = time.time()
        print("Hierarchical (Agglomerative) clustering of X using Scipy library")
        self.hier_scipy_obj = linkage(y=X, method=self.linkage_method, metric=self.dist_metric)
        self.hier_scipy_obj = fcluster(Z=self.hier_scipy_obj, t=self.n_cluster, criterion='maxclust')
        self.runtime[1] = time.time() - start_time
        print("Clustering ended in " + "{0:.2f}".format(round(self.runtime[1], 2)) + " seconds")

    def hierarchy_sklearn_fit(self, X):
        # Record fitting runtime
        start_time = time.time()
        print("Hierarchical (Agglomerative) clustering of X using Sklearn library")
        self.agglo_obj.fit(X)
        self.runtime[2] = time.time() - start_time
        print("Clustering ended in " + "{0:.2f}".format(round(self.runtime[2], 2)) + " seconds")

    def dbscan_fit(self, X):
        # Record fitting runtime
        start_time = time.time()
        print("Clustering X using DBSCAN")
        self.dbscan_obj.fit(X)
        self.runtime[3] = time.time() - start_time
        print("Clustering ended in " + "{0:.2f}".format(round(self.runtime[3], 2)) + " seconds")

    def cluster_fit_all(self, X):
        self.kmeans_fit(X)
        self.hierarchy_scipy_fit(X)
        self.hierarchy_sklearn_fit(X)
        self.dbscan_fit(X)

    def get_cluster_results(self):

        cl_al_dict = {
            "kmeans": self.kmeans_obj.labels_,
            "agnes_scipy": self.hier_scipy_obj,
            "agnes_sklearn": self.agglo_obj.labels_,
            "dbscan": self.dbscan_obj.labels_,
        }

        return cl_al_dict




