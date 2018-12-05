import h5py
import numpy


class readH5:

    gene_names = None
    barcodes = None
    indices = None
    indptr = None
    shape = None
    data = None

    def __init__(self, path):
        fformat = h5py.File(path, 'r')
        temp = fformat['mm10/gene_names'].value
        self.gene_names = []
        for i in range(0, len(temp)):
            self.gene_names.append(temp[i].decode('utf-8'))

        temp = fformat['mm10/barcodes'].value
        self.barcodes = []
        for i in range(0, len(temp)):
            self.barcodes.append(temp[i].decode('utf-8'))

        self.indices = fformat['mm10/indices'].value
        self.indptr = fformat['mm10/indptr'].value
        self.shape = fformat['mm10/shape'].value
        self.data = fformat['mm10/data'].value

        del temp

    def datatoMatrix(self):

        data_matrix = numpy.zeros(self.shape)
        col_count = 0
        for i in range(1, len(self.indptr)):
            sub_indices = self.indices[self.indptr[i-1]:self.indptr[i]]
            sub_data = self.data[self.indptr[i-1]:self.indptr[i]]
            sub_row_count = 0
            for j in sub_indices:
                data_matrix[j, col_count] = sub_data[sub_row_count]
                sub_row_count = sub_row_count + 1
            col_count = col_count + 1
        self.data = data_matrix










