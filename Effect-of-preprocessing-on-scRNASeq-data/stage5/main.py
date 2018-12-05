from SimpleDTLearner import SimpleDTLearner
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import scanpy.api as sc
import numpy, pandas, os, magic.magic
from dca.api import dca
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics.pairwise import euclidean_distances
from readH5 import readH5
from Utility import Utility


def main():

    data_dir = "20KMouseNeurons10X"
    min_samples_split = 1000
    min_samples_leaf = 100

    # Check to see if processed data already exists
    fil_count = os.path.exists("Data/"+data_dir+"/Filtered/filtered_count.csv")
    fil_norm = os.path.exists("Data/"+data_dir+"/Filtered/filtered_normalized.csv")
    dca_fil_norm = os.path.exists("Data/"+data_dir+"/Filtered/dca_transformed.csv")
    mag_fil_norm = os.path.exists("Data/"+data_dir+"/Filtered/magic_transformed.csv")
    cel = os.path.exists("Data/"+data_dir+"/Filtered/cells.csv")

    # Create Utility object
    Uobj = Utility()

    if not fil_count or not fil_norm:
        # Extract the data from the H5 object
        data_matrix, gene_names, sample_names = Read_data(data_dir)

        print(data_matrix.shape)

        # Filter out poor genes and cell samples
        data_matrix, gene_names, sample_names = FilterData(data_matrix, gene_names, sample_names, Uobj)

        # Write the processed Data
        Uobj.WriteData(data_matrix, data_dir, 'filtered_count.csv')

        Uobj.WriteData(gene_names, data_dir, 'genes.csv')

        Uobj.WriteData(sample_names, data_dir, 'sample_names.csv')

        # Normalize the data
        norm_data_matrix = Uobj.NormalizeData(data_matrix=data_matrix)

        # Write normalized data
        Uobj.WriteData(norm_data_matrix, data_dir, 'filtered_normalized.csv')

    # The cell types from the data are unknown but we know that 3 cell types were measured
    # Hence, we cluster the cells based on their similarity
    if not cel:
        # We cluster the cells into 3 types using hclust
        # Read the normalized data
        norm_data_matrix = pandas.read_csv("Data/"+data_dir+"/Filtered/filtered_normalized.csv")

        # Build clusters
        clusters = ClusterCellTypes(norm_data_matrix)

        # Write the results on disk
        Uobj.WriteData(clusters, data_dir, 'cells.csv')

        # Produce PCA
        ProducePCA(norm_data_matrix, clusters, "Original log counts", "_og_log_counts.pdf")

    if not dca_fil_norm:
        # Read the processed count data
        sc_data_matrix = pandas.read_csv("Data/"+data_dir+"/Filtered/filtered_count.csv")

        # Transform with DCA
        sc_data_matrix = DCATransform(sc_data_matrix)

        # Write the transformed data
        Uobj.WriteData(sc_data_matrix, data_dir, 'dca_transformed.csv')

        # Normalize the data
        sc_data_matrix = Uobj.NormalizeData(sc_data_matrix)

        # Read in clustered cells
        cells = pandas.read_csv("Data/"+data_dir+"/Filtered/cells.csv")
        cells = numpy.asarray(cells.iloc[:, 0])

        # Produce PCA
        ProducePCA(sc_data_matrix, cells, "DCA transformed log counts", "_dca_log_counts.pdf")

    if not mag_fil_norm:
        # Read the processed count data
        sc_data_matrix = pandas.read_csv("Data/" + data_dir + "/Filtered/filtered_count.csv")

        # Transform with MAGIC
        sc_data_matrix = MAGICTransform(sc_data_matrix)

        # Write the transformed data
        Uobj.WriteData(sc_data_matrix, data_dir, 'magic_transformed.csv')

        # Normalize the data
        sc_data_matrix = Uobj.NormalizeData(sc_data_matrix.values)

        # Read in clustered cells
        cells = pandas.read_csv("Data/" + data_dir + "/Filtered/cells.csv")
        cells = numpy.asarray(cells.iloc[:, 0])

        # Produce PCA
        ProducePCA(sc_data_matrix, cells, "MAGIC transformed log counts", "_magic_log_counts.pdf")

    # All required data should already be in the required directory
    # Read the processed log normalized data
    norm_data_matrix = pandas.read_csv("Data/"+data_dir+"/Filtered/filtered_normalized.csv")
    print("Noisy log data has " + str(norm_data_matrix.shape[0]) + " genes and " +
          str(norm_data_matrix.shape[1]) + " cells")
    norm_data_matrix = numpy.transpose(norm_data_matrix)

    # Read the DCA transformed log data
    norm_data_matrix_dca = pandas.read_csv("Data/" + data_dir + "/Filtered/dca_transformed.csv")
    print("DCA transformed log data has " + str(norm_data_matrix_dca.shape[0]) + " genes and " +
          str(norm_data_matrix_dca.shape[1]) + " cells")
    norm_data_matrix_dca = numpy.transpose(norm_data_matrix_dca)

    # Read the MAGIC transformed log data
    norm_data_matrix_magic = pandas.read_csv("Data/" + data_dir + "/Filtered/magic_transformed.csv")
    print("MAGIC transformed log data has " + str(norm_data_matrix_magic.shape[0]) + " genes and " +
          str(norm_data_matrix_magic.shape[1]) + " cells")
    norm_data_matrix_magic = numpy.transpose(norm_data_matrix_magic)

    # Read in clustered cells
    cells = pandas.read_csv("Data/" + data_dir + "/Filtered/cells.csv")
    cells = numpy.asarray(cells.iloc[:, 0])

    # Apply the Decision Tree classifier on simple normalized data
    sim_nodec, sim_accu = Evaluate_Data(norm_data_matrix, cells, 1, min_samples_leaf, min_samples_split)

    # Apply the Decision Tree classifier on DCA transformed data
    dca_nodec, dca_accu = Evaluate_Data(norm_data_matrix_dca, cells, 1, min_samples_leaf, min_samples_split)

    # Apply the Decision Tree classifier on MAGIC transformed data
    magic_nodec, magic_accu = Evaluate_Data(norm_data_matrix_magic, cells, 1, min_samples_leaf, min_samples_split)

    # Write accuracy results
    clnm = [str(x) for x in range(1, 11)]
    sim_accu = pandas.DataFrame(sim_accu).T
    sim_accu.columns = clnm

    dca_accu = pandas.DataFrame(dca_accu).T
    dca_accu.columns = clnm
    sim_accu = sim_accu.append(dca_accu, ignore_index=True)

    magic_accu = pandas.DataFrame(magic_accu).T
    magic_accu.columns = clnm
    sim_accu = sim_accu.append(magic_accu, ignore_index=True)

    sim_accu.index = ['Simple', 'DCA', 'MAGIC']

    sim_accu.to_csv("20kMouseNeuron_accuracy_10fold_3meth_results.csv")

    # Write the node count results
    sim_nodec = pandas.DataFrame(sim_nodec).T
    sim_nodec.columns = clnm

    dca_nodec = pandas.DataFrame(dca_nodec).T
    dca_nodec.columns = clnm
    sim_nodec = sim_nodec.append(dca_nodec, ignore_index=True)

    magic_nodec = pandas.DataFrame(magic_nodec).T
    magic_nodec.columns = clnm
    sim_nodec = sim_nodec.append(magic_nodec, ignore_index=True)

    sim_nodec.index = ['Simple', 'DCA', 'MAGIC']

    sim_nodec.to_csv("20kMouseNeuron_nodecount_10fold_3meth_results.csv")


def Read_data(data_dir):

    # Read h5 object
    h5obj = readH5(path="./Data/"+data_dir+"/Raw/1M_neurons_neuron20k.h5")
    h5obj.datatoMatrix()
    data_matrix = h5obj.data
    gene_names = h5obj.gene_names
    sample_names = h5obj.barcodes

    return data_matrix, gene_names, sample_names


def FilterData(data_matrix, gene_names, sample_names, Uobj):

    # Remove genes with low dispersion
    data_matrix, gene_names = Uobj.filterGenes(data_matrix=data_matrix, gene_names=gene_names,
                                               data_name='sc_20k_mouse_neurons')
    # Remove cells with low feature count
    data_matrix, sample_names = Uobj.filterCells(data_matrix=data_matrix, sample_names=sample_names,
                                                 data_name='sc_20k_mouse_neurons')

    return data_matrix, gene_names, sample_names


def ClusterCellTypes(norm_data_matrix):

    # Compute pairwise distance between cells
    clust_dist = euclidean_distances(X=numpy.transpose(norm_data_matrix))
    # Build hclust dendogram
    clust_link = linkage(clust_dist, 'ward')
    # Cut at 3 clusters
    clusters = fcluster(Z=clust_link, t=3, criterion='maxclust')

    return clusters


def DCATransform(sc_data_matrix):

    # Create a scanpy AnnData object
    sc_data_matrix = sc.AnnData(numpy.transpose(sc_data_matrix.values))

    # Filter genes with count<2
    sc.pp.filter_genes(data=sc_data_matrix, min_counts=1)

    # Apply DCA transform
    dca(adata=sc_data_matrix, threads=4, epochs=10)

    print("DCA Denoised data prepared")

    return numpy.transpose(sc_data_matrix.X)


def MAGICTransform(sc_data_matrix):

    # Apply MAGIC imputation
    sc_data_matrix = magic.MAGIC().fit_transform(sc_data_matrix)

    print("MAGIC data prepared")

    return sc_data_matrix


def ProducePCA(norm_data_matrix, cells, type, fig_name):

    # Convert cell types to string
    cells = ["cell type " + str(c) for c in cells]

    # Create a scanpy object
    sc_data_matrix_og = sc.AnnData(numpy.transpose(norm_data_matrix))

    # Assign observation name
    sc_data_matrix_og.obs['cells'] = cells
    sc_data_matrix_og.obs_names = cells

    # Perform PCA with 2 dimensions
    sc.pp.pca(sc_data_matrix_og, n_comps=2)

    # Plot results
    sc.pl.pca_scatter(sc_data_matrix_og, color='cells', title=type, save=fig_name)


def Evaluate_Data(X, Y, random_state, min_samples_leaf, min_samples_split):

    X = StandardScaler().fit_transform(X)

    k_fold = KFold(n_splits=10, random_state=random_state)
    accu = []
    node_count = []
    lcty = SimpleDTLearner(random_state=random_state,
                           min_samples_leaf=min_samples_leaf,
                           min_samples_split=min_samples_split)

    for train, test in k_fold.split(X):
        X_train, X_test = X[train, :], X[test, :]
        Y_train, Y_test = Y[train], Y[test]
        lcty.FitDTLearn(X=X_train, Y=Y_train)
        Y_pred = lcty.PredictDTLearn(X_test)
        acc = (numpy.where(Y_pred != Y_test, 0, 1)).mean()
        accu.append(acc)
        node_count.append(lcty.dt_learn.tree_.node_count)

    return node_count, accu


main()
