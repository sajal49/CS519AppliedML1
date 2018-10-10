from ClassifierPool import ClassifierPool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import argparse
import os
from progress.bar import Bar
import numpy
import pandas


def main():
    # setup args list
    arguments = setup_arguments()
    args = arguments.parse_args()

    # check parameters
    if args.data_to_analyze is None:
        print("No data to fit! please specify an option for -data_to_analyze, check python3 main.py --help for "
              "options. Exiting..")
        raise SystemExit
    if args.data_to_analyze == '2' and args.data_path is None:
        print("Option '2' for -data_to_analyze requires -data_path (root) ! check python3 main.py --help for "
              "options. Exiting..")
        raise SystemExit
    if args.data_to_analyze == '3' and (args.data_path is None or args.data_name is None):
        print("Option '3' for -data_to_analyze requires -data_path and -data_name! check python3 main.py --help for "
              "options. Exiting..")
        raise SystemExit
    if not (args.data_to_analyze == '1' or args.data_to_analyze == '2' or args.data_to_analyze == '3'):
        print("-data_to_analyze can be 1, 2, or 3. Check python3 main.py --help for options. Exiting..")
        raise SystemExit

    # select data to evaluate
    if args.data_to_analyze == '2':
        X, Y = Fetch_Activity_Recognition(args.data_path)
        data_name = "Activity_Recognition"
    elif args.data_to_analyze == '1':
        X, Y = Fetch_Digits()
        data_name = "Digits"
    else:
        data = pandas.read_csv(args.data_path)
        # Extract X
        X = data.iloc[:, 0:(data.shape[1] - 1)]
        # Extract Y
        Y = data.iloc[:, (data.shape[1] - 1)]
        data_name = args.data_name

    # Evaluate data
    result = Evaluate_Data(X, Y, args)

    # Save results
    result.to_csv(data_name+"_results.csv")


def Fetch_Digits():
    data = load_digits()
    X = pandas.DataFrame(data['data'])
    Y =data['target']
    return X, Y


def Evaluate_Data(X, Y, args):

    # Scale X
    X_scaled = StandardScaler().fit_transform(X=X)

    # Split data for cross validation
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, train_size=0.7,
                                                        random_state=1 if args.seed is None
                                                        else int(args.seed), stratify=Y)
    # Build the classifier pool
    cp = ClassifierPool(n_jobs=4 if args.n_jobs is None else int(args.n_jobs),
                        max_iter=10 if args.max_iter is None else int(args.max_iter),
                        random_state=1 if args.seed is None else int(args.seed),
                        dt_min_split=10 if args.dt_min_split is None else int(args.dt_min_split),
                        lsvm_penalty='l2' if args.lsvm_penalty is None else args.lsvm_penalty,
                        lsvm_c=0.05 if args.lsvm_c is None else float(args.lsvm_c),
                        nlsvm_c=0.05 if args.nlsvm_c is None else float(args.nlsvm_c),
                        nlsvm_gma='auto' if args.nlsvm_gma is None else float(args.nlsvm_gma),
                        ptron_penalty='l2' if args.ptron_penalty is None else args.ptron_penalty,
                        ptron_c=0.05 if args.ptron_c is None else float(args.ptron_c),
                        ptron_eta=0.001 if args.ptron_eta is None else float(args.ptron_eta),
                        logres_penalty='l2' if args.logres_penalty is None else args.logres_penalty,
                        logres_c=0.05 if args.logres_c is None else float(args.logres_c),
                        knn_k=3 if args.knn_k is None else int(args.knn_k),
                        knn_algo='kd_tree' if args.knn_algo is None else args.knn_algo)

    # Learn training data
    cp.fit_all_classifier(X_train, Y_train)

    print("Models learned..")

    # Use the learned model to predict Y for test samples
    cp_predict_dict_test = cp.predict_all_classifier(X_test)

    print("Prediction on testing data done..")

    # Use the learned model to predict Y for test samples
    cp_predict_dict_train = cp.predict_all_classifier(X_train)

    print("Prediction on training data done..")

    # Classifier list
    cl_list = ["dt", "lsvm", "nlsvm", "ptron", "logres", "knn"]

    # Runtime for building model
    result = pandas.DataFrame(cp.runtime).T
    result.columns = cl_list

    # Prediction on test data
    accu_test = []
    for cl in cl_list:
        Y_pred = cp_predict_dict_test[cl]
        accu = numpy.where(Y_pred != Y_test, 0, 1)
        accu = accu.mean()
        accu_test.append(accu)

    accu_test = pandas.DataFrame(accu_test).T
    accu_test.columns = cl_list
    result = result.append(accu_test, ignore_index=True)

    # Predictions on training data
    accu_train = []
    for cl in cl_list:
        Y_pred = cp_predict_dict_train[cl]
        accu = numpy.where(Y_pred != Y_train, 0, 1)
        accu = accu.mean()
        accu_train.append(accu)

    accu_train = pandas.DataFrame(accu_train).T
    accu_train.columns = cl_list
    result = result.append(accu_train, ignore_index=True)
    result.index = ['runtime', 'pred on test', 'pred on training']

    return result


def Fetch_Activity_Recognition(root):

    # Handling the annoying .DS_Store file
    if os.path.exists(root+".DS_Store"):
        os.remove(root+".DS_Store")

    files = os.listdir(root)

    # Construct column names
    Modalities = ["ACC : X", "ACC : Y", "ACC : Z", "GYR : X", "GYR : Y", "GYR : Z", "MAG : X", "MAG : Y", "MAG : Z",
                  "QUAT : 1", "QUAT : 2", "QUAT : 3", "QUAT : 4"]
    Sensors = ["RLA", "RUA", "BACK", "LUA", "LLA", "RC", "RT", "LT", "LC"]
    sens_x_modal = []
    for sens in Sensors:
        for mods in Modalities:
            sens_x_modal.append(sens+" - "+mods)
    col_names = []
    col_names = col_names + ["Time_stamp1"] + ["Time_stamp2"] + sens_x_modal + ["Activity type"]

    data = pandas.read_table(filepath_or_buffer=root+files[0], sep="\t")
    data.columns = col_names

    pb = Bar("Files", max=(len(files)-1))
    for file in files[1:]:
        tble = pandas.read_table(filepath_or_buffer=root+file, sep="\t")
        tble.columns = col_names
        data = data.append(other=tble, ignore_index=True, sort=False)
        pb.next()

    pb.finish()

    # Extract X
    X = data.iloc[:, 0:(data.shape[1] - 1)]
    # Extract Y
    Y = data.iloc[:, (data.shape[1] - 1)]

    return X, Y


# setup argument names and help options
def setup_arguments():

    arguments = argparse.ArgumentParser()
    arguments.add_argument('-seed', help='int; Random seed.')
    arguments.add_argument('-n_jobs', help='int; Number of parallel threads to be used.')
    arguments.add_argument('-max_iter', help='int; Maximum number of epochs for Perceptron and SVM classifiers.')
    arguments.add_argument('-dt_min_split', help='int; Decision Tree : The minimum number of samples required to split '
                                                 'an internal node. (default = 10)')
    arguments.add_argument('-lsvm_penalty', help='string; Linear SVM : Penalty of l1 or l2 (default = l2)')
    arguments.add_argument('-lsvm_c', help='float; Linear SVM : Penalty parameter C of the error term (default = 0.05)')
    arguments.add_argument('-nlsvm_c', help='float; Non-Linear SVM : Penalty parameter C of the error term '
                                            '(default = 0.05)')
    arguments.add_argument('-nlsvm_gma', help='float; Non-Linear SVM : Kernel coefficient (default = 1/n_attributes)')
    arguments.add_argument('-ptron_penalty', help='string; Perceptron : Penalty of l1, l2 or elasticnet (default = l2)')
    arguments.add_argument('-ptron_c', help='float; Perceptron : Multiplies the penalty term  (default = 0.05)')
    arguments.add_argument('-ptron_eta', help='double; Perceptron : Constant by which the updates are multiplied '
                                              '(default = 0.001)')
    arguments.add_argument('-logres_penalty', help='string; Logistic Reg. : Penalty of l1 or l2 (default = l2)')
    arguments.add_argument('-logres_c', help='float; Logistic Reg. : Penalty parameter C of the error term '
                                             '(default = 0.05)')
    arguments.add_argument('-knn_k', help='int; KNN Class. : Number of neighbors to use  (default = 3)')
    arguments.add_argument('-knn_algo', help='string; KNN Class. : Chose from auto, ball_tree, kd_tree, brute '
                                             '(default = kd_tree)')
    arguments.add_argument('-data_to_analyze', help='int; Proj 3 evaluates on 2 datasets, 1) digits and 2) activity'
                                                    'recognition, if a third data set has to be evaluated please'
                                                    'set this option to 3 and provide a -data_path and -data_name. If'
                                                    'option 2 is being used, please supply a path to all individual'
                                                    'files to merge. Option 2 can take forever to run so it is '
                                                    'suggested to use at-most 5 files from the data set.[*]')
    arguments.add_argument('-data_path', help='Path to the data file; only valid if -data_to_analyze=3. The last '
                                              'column will be treated as class labels and the rest as attributes.')
    arguments.add_argument('-data_name', help='Name of the data-set being evaluated; only valid if -data_to_analyze=3')
    return arguments

main()