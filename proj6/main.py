import numpy
from DimensionalityReductionPool import DimensionalityReductionPool
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse
import matplotlib.pyplot as plt
import pandas
import copy


def main():
    # setup args list
    arguments = setup_arguments()
    args = arguments.parse_args()

    # check parameters
    if args.data_to_analyze == 3 and args.data_path is None:
        print("Option 3 for -data_to_analyze requires -data_path and -data_name! check python3 main.py --help for "
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
        X, Y = FetchDigits()
        data_name = "Digits"
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
        Y = data.iloc[:, (data.shape[1] - 1)]

    # Get result
    result = Evaluate_Data(X, Y, args, data_name)

    # Save result
    result.to_csv(data_name + "_results.csv")


def Evaluate_Data(X, Y, args, data_name):

    # Standardize the data set
    X = StandardScaler().fit_transform(X=X)

    # Split the data into training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, stratify=Y)

    # Prepare logistic regression obj
    Logit_obj = LogisticRegression(fit_intercept=False, random_state=args.seed, solver='sag', multi_class='ovr',
                                   n_jobs=args.n_jobs)

    # Build the dimensionality reduction pool
    if args.gamme_kernel_pca == 0:
        gamma = 1/(X.shape[1])
    else:
        gamma = args.gamma_kernel_pca

    dimredpool = DimensionalityReductionPool(random_state=args.seed,
                                             n_jobs=args.n_jobs,
                                             gamme_kernel_pca=gamma)
    # Fit training data
    dimredpool.Fit_all(X_train, Y_train)

    # Transform training data
    transformed_x_dict = dimredpool.Transform_all(X_train)

    print("Data Transformed!!")

    # Get cumulative variance
    var_explained_dict = GetVarianceExplained(dimredpool, transformed_x_dict)

    # Plot top two components
    Plot_top_two_components(transformed_x_dict, var_explained_dict, Y_train, data_name)

    # Choose attributes from transformed X that satisfies -var_explained
    keys = list(var_explained_dict.keys())
    filtered_indx = []
    for k in keys:
        cum_var = var_explained_dict[k]
        indx = numpy.sort(sum(numpy.where(cum_var >= args.var_explained)))[0]
        transformed_x_dict[k] = transformed_x_dict[k][:, 0:(indx+1)]
        filtered_indx.append(indx+1)

    # Runtime for building model
    result = pandas.DataFrame(dimredpool.runtime).T
    result.columns = keys

    # Apply logistic regression on the transformed X
    print("Applying Logistic regression on all transformed data")
    logit_obj_pool = []
    for k in keys:
        logit_obj_pool.append(copy.deepcopy(Logit_obj))

    accuracy_train = []
    accuracy_test = []
    for i in range(0, len(keys)):
        logit_obj_pool[i].fit(X=transformed_x_dict[keys[i]], y=Y_train)
        pred_train = logit_obj_pool[i].predict(transformed_x_dict[keys[i]])
        accuracy_train.append(numpy.where(pred_train != Y_train, 0, 1).mean())

        transformed_x_test = dimredpool.GetDimRedObjects()[keys[i]].transform(X_test)
        transformed_x_test = transformed_x_test[:, 0:(filtered_indx[i])]
        pred_test = logit_obj_pool[i].predict(transformed_x_test)
        accuracy_test.append(numpy.where(pred_test != Y_test, 0, 1).mean())

    print("Data model learned..")

    accuracy_train = pandas.DataFrame(accuracy_train).T
    accuracy_train.columns = keys
    result = result.append(accuracy_train, ignore_index=True)

    accuracy_test = pandas.DataFrame(accuracy_test).T
    accuracy_test.columns = keys
    result = result.append(accuracy_test, ignore_index=True)

    filtered_indx = numpy.asarray(filtered_indx)/X_train.shape[1]
    filtered_indx = pandas.DataFrame(filtered_indx).T
    filtered_indx.columns = keys
    result = result.append(filtered_indx, ignore_index=True)

    result.index = ['runtime', 'pred on test', 'pred on training', 'dimension usage']

    return result


def GetVarianceExplained(dimredpool, transformed_x_dict):

    kernelpca_ex_var = numpy.var(transformed_x_dict['KernelPCA'], axis=0)
    variance_explained_dict = {
        'PCA': numpy.cumsum(dimredpool.pca_obj.explained_variance_ratio_),
        'LDA': numpy.cumsum(dimredpool.lda_obj.explained_variance_ratio_),
        'KernelPCA': numpy.cumsum(kernelpca_ex_var/numpy.sum(kernelpca_ex_var))
    }

    return variance_explained_dict


def Plot_top_two_components(transformed_x_dict, variance_exp_dict, Y, data_name):

    keys = list(transformed_x_dict.keys())
    uny = numpy.unique(Y)
    for i in range(0, len(transformed_x_dict)):
        fig = plt.figure()
        x_transformed = transformed_x_dict[keys[i]]
        for j in range(0, len(uny)):
            plt.scatter(x_transformed[Y == uny[j], 0], x_transformed[Y == uny[j], 1],
                        label=uny[j])
            plt.xlabel(keys[i] + "-1")
            plt.ylabel(keys[i] + "-2")
        plt.legend(loc='best', shadow=False)
        plt.title(data_name + " " + keys[i] + " transformation" + "\nVariance Explained : " +
                  "{0:.2f}".format(round(variance_exp_dict[keys[i]][1], 2)))
        fig.savefig(data_name + "_" + keys[i] + "_top_two_components_.pdf", bbox_inches='tight')


def FetchIris():

    data = load_iris()
    X = data['data']
    Y = data['target']
    return X, Y


def FetchDigits():

    data = load_digits()
    X = data['data']
    Y = data['target']
    return X, Y


# setup argument names and help options
def setup_arguments():

    arguments = argparse.ArgumentParser()

    arguments.add_argument('-seed', help='int; Random seed. (Default: 1)', default=1, required=False, type=int)
    arguments.add_argument('-n_jobs', help='int; Number of parallel threads to be used. (Default: 4)', default=4,
                           required=False, type=int)
    arguments.add_argument('-var_explained',help='float; Cutoff for variance explained. Will be used to determine '
                                                 'n_components. (Default: 0.9)', type=float, default=0.9,
                           required=False)
    arguments.add_argument('-data_to_analyze', help='int; Proj 6 evaluates on 2 datasets, 1) Iris and 2) Digits'
                                                    'If a third data set has to be evaluated please set this '
                                                    'option to 3 and provide a -data_path, -data_name.[*]'
                                                    'NOTE : This implementation only supports data-sets with'
                                                    'categorical Y.', type=int, required=True)
    arguments.add_argument('-data_path', help='Path to the data file; valid if -data_to_analyze is set to 3. The last '
                                              'column will be treated as response and the rest as attributes [*]',
                           type=str)
    arguments.add_argument('-data_name', help='Name of the data-set being evaluated; valid if -data_to_analyze is set'
                                              'to 3', type=str)
    arguments.add_argument('-gamme_kernel_pca', help='float; Gamma for RBF Kernel PCA. (Default: 1 / n_features)',
                           default=0.0, required=False, type=float)
    return arguments

main()