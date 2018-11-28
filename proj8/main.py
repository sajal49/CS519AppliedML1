import numpy
from EnsemblePool import EnsemblePool
from sklearn.datasets import load_digits
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse
import pandas


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
        X, Y = FetchDigits()
        data_name = "Digits"
    elif args.data_to_analyze == 2:
        X, Y = FetchMMass()
        data_name = "MMass"
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
    result = Evaluate_Data(X, Y, args)

    # Save result
    result.to_csv(data_name + "_results.csv")


def Evaluate_Data(X, Y, args):

    # Standardize the data set
    X = StandardScaler().fit_transform(X=X)

    # Split the data into training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, stratify=Y, random_state=args.seed)

    ensemblepool = EnsemblePool(random_state=args.seed,
                                n_jobs=args.n_jobs,
                                n_estimators=args.n_estimators,
                                min_samples_leaf=args.min_samples_leaf)
    # Fit training data
    ensemblepool.Fit_all(X_train, Y_train)

    print("Models learned!!")

    # Classifier list
    cl_list = ['Base DT', 'ADAB DT', 'RF', 'BAG DT']

    # Runtime for building model
    runtime = ensemblepool.runtime
    result = pandas.DataFrame(runtime).T
    result.columns = cl_list

    # Predictions on training data
    print("Predicting training data!!")
    accu_train = []
    ensemble_predict_dict = ensemblepool.Predict_all(X_train)
    for cl in cl_list:
        Y_pred = ensemble_predict_dict[cl]
        accu = numpy.where(Y_pred != Y_train, 0, 1)
        accu = accu.mean()
        accu_train.append(accu)

    accu_train = pandas.DataFrame(accu_train).T
    accu_train.columns = cl_list
    result = result.append(accu_train, ignore_index=True)

    # Prediction on test data
    print("Predicting test data!!")
    accu_test = []
    ensemble_predict_dict = ensemblepool.Predict_all(X_test)
    for cl in cl_list:
        Y_pred = ensemble_predict_dict[cl]
        accu = numpy.where(Y_pred != Y_test, 0, 1)
        accu = accu.mean()
        accu_test.append(accu)

    accu_test = pandas.DataFrame(accu_test).T
    accu_test.columns = cl_list
    result = result.append(accu_test, ignore_index=True)
    result.index = ['runtime', 'pred on train', 'pred on test']

    return result


def FetchMMass():

    data = pandas.read_csv('Dataset/mammographic_masses.data', header=None, na_values='?')
    simp = SimpleImputer(missing_values=numpy.NAN, strategy='median')
    data = simp.fit_transform(X=data.values)
    X = data[:, 0:(data.shape[1] - 1)]
    Y = data[:, data.shape[1] - 1]
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
    arguments.add_argument('-n_estimators', help='int; The maximum number of estimators for the ensemble methods '
                                                 '(Default:10).', type=int, default=10, required=False)
    arguments.add_argument('-min_samples_leaf', help='int; The minimum number of samples required to be at a leaf node '
                                                     '(Default: 10).', required=False, type=int, default=10)
    arguments.add_argument('-data_to_analyze', help='int; proj 8 evaluates on 2 datasets, 1) Digits and 2) Mammographic'
                                                    'Mass. If a third data set has to be evaluated please set this '
                                                    'option to 3 and provide a -data_path, -data_name.[*]'
                                                    'NOTE : This implementation only supports data-sets with'
                                                    'categorical Y.', type=int, required=True)
    arguments.add_argument('-data_path', help='Path to the data file; valid if -data_to_analyze is set to 3. The last '
                                              'column will be treated as response and the rest as attributes [*]',
                           type=str)
    arguments.add_argument('-data_name', help='Name of the data-set being evaluated; valid if -data_to_analyze is set'
                                              ' to 3', type=str)
    return arguments

main()