import numpy, os, pandas
import matplotlib.pyplot as plt


class Utility:

    def __init__(self):
        pass

    def scMAD(self, x):
        med_x = numpy.median(x)
        k = 1/(numpy.percentile(x, 75))
        return k * numpy.median(abs(x - med_x))

    def filterGenes(self, data_matrix, gene_names, data_name):

        gene_mean = numpy.mean(data_matrix, axis=1)
        keep_genes = sum(numpy.where(gene_mean != 0))

        data_matrix = data_matrix[keep_genes, :]
        gene_names = numpy.asanyarray(gene_names)
        gene_names = gene_names[keep_genes]
        gene_mean = gene_mean[keep_genes]

        gene_std = numpy.std(data_matrix, axis=1)
        coeff_of_var = gene_std / gene_mean

        fig = plt.figure(1)
        plt.hist(coeff_of_var, bins='auto', color='blue', histtype='stepfilled', alpha=0.5)
        plt.xlabel('Coefficient of variation in genes (Dispersion around mean)')
        plt.ylabel('Count frequency')

        med_cov = numpy.median(coeff_of_var)
        mad_cov = self.scMAD(coeff_of_var)
        threshold = med_cov - 3 * mad_cov

        plt.axvline(x=threshold, linestyle='-.', color='red')
        fig.savefig('figures/' + data_name + '_coef_of_variation_cutoff' + '.png', bbox_inches='tight', dpi=1000)
        plt.close(fig)

        keep_genes = sum(numpy.where(coeff_of_var>=threshold))
        data_matrix = data_matrix[keep_genes, :]
        gene_names = gene_names[keep_genes]

        return data_matrix, gene_names

    def filterCells(self, data_matrix, sample_names, data_name):

        sample_feature_count = numpy.zeros(data_matrix.shape[1])
        for i in range(0, data_matrix.shape[1]):
            sample_feature_count[i] = numpy.log1p(numpy.sum(numpy.where(data_matrix[:, i] != 0, 1, 0)))

        fig = plt.figure(1)
        plt.hist(sample_feature_count, bins='auto', color='blue', histtype='stepfilled', alpha=0.5)
        plt.xlabel('Log Feature count')
        plt.ylabel('Count frequency')

        med_cov = numpy.median(sample_feature_count)
        mad_cov = self.scMAD(sample_feature_count)
        threshold = med_cov - 3 * mad_cov

        plt.axvline(x=threshold, linestyle='-.', color='red')
        fig.savefig('figures/' + data_name + '_sample_feature_count_cutoff' + '.png', bbox_inches='tight', dpi=1000)
        plt.close(fig)

        keep_samples = sum(numpy.where(sample_feature_count >= threshold))
        data_matrix = data_matrix[:, keep_samples]
        sample_names = numpy.asanyarray(sample_names)
        sample_names = sample_names[keep_samples]

        return data_matrix, sample_names

    def NormalizeData(self, data_matrix):

        umi_count = numpy.sum(data_matrix, axis=0)
        md_umi_count = numpy.median(umi_count)

        for i in range(0, data_matrix.shape[1]):
            data_matrix[:, i] = numpy.log1p((data_matrix[:, i] * md_umi_count)/umi_count[i])

        return data_matrix

    def WriteData(self, data_matrix, data_dir, file_name):

        data_matrix = pandas.DataFrame(data=data_matrix)
        store_dir = "Data/"+data_dir+"/Filtered/"
        if not os.path.exists(store_dir):
            os.makedirs(store_dir)
        data_matrix.to_csv(path_or_buf=store_dir + "/" + file_name, index=False)
        print('File ' + store_dir + "/" + file_name + " was written")

    def LogTransformCell(self, data_matrix):

        for i in range(0, data_matrix.shape[1]):
            data_matrix[:, i] = numpy.log1p(data_matrix[:, i])

        return data_matrix