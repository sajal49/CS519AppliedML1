import numpy
from ClusteringAlgorithmPool import ClusteringAlgorithmPool
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from ClustUtility import ClustUtility
from scipy.stats import chi2_contingency
import argparse
import pandas


def main():
    # setup args list
    arguments = setup_arguments()
    args = arguments.parse_args()

    # check parameters
    if (args.data_to_analyze == 2 or args.data_to_analyze == 3) and args.data_path is None:
        print("Option 2, 3 for -data_to_analyze requires -data_path and -data_name! check python3 main.py --help for "
              "options. Exiting..")
        raise SystemExit
    if not (1 <= args.data_to_analyze <= 3):
        print("-data_to_analyze is an integer between [1,3]. Check python3 main.py --help for options. Exiting..")
        raise SystemExit
    # select data to evaluate
    if args.data_to_analyze == 1:
        X, Y = FetchIris()
        data_name = "Iris"
    elif args.data_to_analyze == 2:
        X, Y = FetchFaultSteePlates(args.data_path)
        data_name = "Fault_Steel_Plates"
    else:
        data = pandas.read_csv(args.data_path)
        # check for non-integer / non-float values
        dt_dtypes = data.dtypes
        dt_dtypes = (dt_dtypes == int) | (dt_dtypes == float)
        for i in range(0, data.shape[1]):
            if not dt_dtypes[i]:
                old_entries = numpy.unique(data.iloc[:, i])
                data.iloc[:, i] = data.iloc[:, i].replace(to_replace=old_entries,
                                                          value=list(range(0, len(old_entries))))
        data_name = args.data_name
        X = data.iloc[:, 0:(data.shape[1] - 1)]
        if not args.no_y:
            Y = data.iloc[:, (data.shape[1] - 1)]
        else:
            Y = []

    # Standardize the data set
    X = StandardScaler().fit_transform(X=X)

    # Get result
    result = Evaluate_Data(X, Y, args, data_name)

    # Save result
    result.to_csv(data_name + "_results.csv")


def Evaluate_Data(X, Y, args, data_name):

    # Build the cluster utility object
    cu = ClustUtility()

    # Build the clustering algorithm pool
    if args.n_cluster == 'auto':
        k = cu.Estimate_K_elbow(X=X, data_name=data_name)
    else:
        k = int(args.n_cluster)

    if args.eps_dbscan == 'auto' or args.min_pts_dbscan == 'auto':
        min_pt_d, eps_d = cu.Estimate_eps_n_min_pts(X=X, K=5, data_name=data_name)

    if args.eps_dbscan != 'auto':
        eps_d = int(args.eps_dbscan)

    if args.min_pts_dbscan != 'auto':
        min_pt_d = int(args.min_pts_dbscan)

    clalpool = ClusteringAlgorithmPool(n_cluster=k,
                                       random_state=args.seed,
                                       n_jobs=args.n_jobs,
                                       dist_metric=args.dist_metric,
                                       linkage_method=args.linkage_method,
                                       eps_dbscan=eps_d,
                                       min_pts_dbscan=min_pt_d)

    # Fit training data
    clalpool.cluster_fit_all(X)

    print("Data clustered!!")

    clal_res = clalpool.get_cluster_results()

    # Clustering algo list
    cl_al_list = ["kmeans", "agnes_scipy", "agnes_sklearn", "dbscan"]

    # Runtime for clustering
    result = pandas.DataFrame(clalpool.runtime).T
    result.columns = cl_al_list

    # SSE for cluster quality
    sse_cl = []

    for cl_al in cl_al_list:
        cluster = clal_res[cl_al]
        centers = cu.ComputeCenters(cluster, X)
        sse_cl.append(cu.ComputeSSE(cluster, centers, X))

    sse_cl = pandas.DataFrame(sse_cl).T
    sse_cl.columns = cl_al_list
    result = result.append(sse_cl, ignore_index=True)
    result.index = ['runtime', 'SSE']

    # Quality of clustering
    if not args.no_y:
        chisq_cl = []
        for cl_al in cl_al_list:
            cluster = clal_res[cl_al]
            obs_table = pandas.crosstab(cluster, Y)
            chisq_cl.append(chi2_contingency(obs_table)[1])
        chisq_cl = pandas.DataFrame(chisq_cl).T
        chisq_cl.columns = cl_al_list
        result = result.append(chisq_cl, ignore_index=True)
        result.index = ['runtime', 'SSE', 'chisq']

    return result


def FetchIris():

    data = load_iris()
    X = data['data']
    Y = data['target']
    return X, Y


def FetchFaultSteePlates(path):

    data = pandas.read_csv(path)
    X = data.iloc[:, 0:27]
    Y = data.iloc[:, 27:34]

    # merge Y
    Y_lab = []
    nm = Y.columns.values.tolist()
    for i in range(0, Y.shape[0]):
        Y_lab.append(nm[sum(numpy.where(Y.iloc[i, :] == 1))[0]])

    return X, Y_lab


# setup argument names and help options
def setup_arguments():

    arguments = argparse.ArgumentParser()

    arguments.add_argument('-seed', help='int; Random seed. (Default: 1)', default=1, required=False, type=int)
    arguments.add_argument('-n_jobs', help='int; Number of parallel threads to be used. (Default: 4)', default=4,
                           required=False, type=int)
    arguments.add_argument('-n_cluster', help='int; Number of clusters. If no number is supplied, Elbow method is used'
                                              'to detect the number of clusters. (Default: auto)', default='auto',
                           required=False)
    arguments.add_argument('-data_to_analyze', help='int; Proj 5 evaluates on 2 datasets, 1) Iris and 2) Fault steel'
                                                    'plates. If a third data set has to be evaluated please set this '
                                                    'option to 3 and provide a -data_path, -data_name.[*]', type=int,
                           required=True)
    arguments.add_argument('-data_path', help='Path to the data file; valid if -data_to_analyze is set to 2 or 3. '
                                              'The last column will be treated as response and the rest as attributes,'
                                              'unless -no_y is used [*]', type=str)
    arguments.add_argument('-data_name', help='Name of the data-set being evaluated; valid if -data_to_analyze is set'
                                              'to 3', type=str)
    arguments.add_argument('-dist_metric', help='string; Distance metric to be used. (Default:euclidean). Options : '
                                                '{“euclidean”, “manhattan”, “cosine”}', default='euclidean',
                           required=False, type=str)
    arguments.add_argument('-linkage_method', help='string; Method for calculating the distance between the newly '
                                                   'formed clusters in Hierarchical clustering. (Default:ward).'
                                                   'Options : {“ward”, “complete”, “average”, “single”}', default='ward'
                           , required=False, type=str)
    arguments.add_argument('-eps_dbscan', help='float; The maximum distance between two samples for them to be '
                                               'considered as in the same neighborhood for DBSCAN (Default:auto).',
                           default='auto',required=False)
    arguments.add_argument('-min_pts_dbscan', help='int; The number of samples (or total weight) in a neighborhood for '
                                                   'a point to be considered as a core point. This includes the point '
                                                   'itself (Default:auto).', default='auto', required=False)
    arguments.add_argument('-no_y', help='bool; Should the last column of data be treated as y? ', const=True,
                           required=False, type=bool, nargs='?')
    return arguments

main()