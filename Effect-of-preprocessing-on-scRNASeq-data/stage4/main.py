import load_single_cell_data as lscd
from LearnCellType import LearnCellType
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import numpy


def main():
    scdata, genes, cells = lscd.read_single_cell_data(root="",
                               gene_file_name="data/genes_hnc_non_cancer.txt",
                               cell_type_file_name="data/cell_type_hnc_non_cancer.txt",
                               scdata_file_name="data/hnc_data_non_cancer.csv")

    scdata = scdata.T
    #scdata = scdata.values
    scdata = StandardScaler().fit_transform(scdata)

    print(scdata.shape)

    k_fold = KFold(n_splits=10, random_state=1)

    accu = []

    lcty = LearnCellType(random_state=1)

    for train, test in k_fold.split(scdata):
        X_train, X_test = scdata[train, :], scdata[test, :]
        Y_train, Y_test = numpy.array(cells)[train], numpy.array(cells)[test]

        lcty.FitDTLearn(X=X_train,Y=Y_train)
        Y_pred = lcty.PredictDTLearn(X_test)

        acc = (numpy.where(Y_pred != Y_test, 0, 1)).mean()
        accu.append(acc)

    print(accu)

main()
