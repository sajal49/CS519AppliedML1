import pandas


def read_single_cell_data(root, gene_file_name, cell_type_file_name, scdata_file_name) :

    # Read data 
    scData = pandas.read_csv(root+scdata_file_name)

    # Read gene names
    with open(root+gene_file_name) as file:
        gene_names = file.readlines()

    gene_names = [line.strip() for line in gene_names]

    # Read cell-types
    with open(root+cell_type_file_name) as file:
        cell_types = file.readlines()

    cell_types = [line.strip() for line in cell_types]

    return scData, gene_names, cell_types