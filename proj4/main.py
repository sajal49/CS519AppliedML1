from RegressorPool import RegressorPool
import pandas
import numpy
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse


def main():
    # setup args list
    arguments = setup_arguments()
    args = arguments.parse_args()

    # check parameters
    if args.data_to_analyze == 4 and args.data_name is None:
        print("Option '4' for -data_to_analyze requires -data_path and -data_name! check python3 main.py --help for "
              "options. Exiting..")
        raise SystemExit
    if not (1 <= args.data_to_analyze <= 4):
        print("-data_to_analyze is an integer between [1,4]. Check python3 main.py --help for options. Exiting..")
        raise SystemExit

    # select data to evaluate
    if args.data_to_analyze == 1:
        data = FetchHousePricing(args.data_path)
        data_name = "House Pricing"
    elif args.data_to_analyze == 2 or args.data_to_analyze == 3:
        solar_energy, wind_energy = FetchCaliforniaRenEnergy(args.data_path)
        if args.data_to_analyze == 2:
            data = solar_energy
            data_name = "Solar Energy"
        else:
            data = wind_energy
            data_name = "Wind Energy"
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

    # Standardize the data set
    data = StandardScaler().fit_transform(X=data)

    # Extract features (independent variables)
    X = data[:, 0:(data.shape[1] - 1)]

    # Extract response
    Y = data[:, (data.shape[1] - 1)]

    # Get result
    result = Evaluate_Data(X, Y, args)

    # Save result
    result.to_csv(data_name + "_results.csv")


def Evaluate_Data(X, Y, args):

    # Split data for cross validation
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7,
                                                        random_state=args.seed)
    # Build the regressor pool
    reg = RegressorPool(random_state=args.seed,
                        n_jobs=args.n_jobs,
                        max_iter=args.max_iter,
                        min_samples_ransac=args.min_samples_ransac,
                        lambda_l1=args.lambda_l1,
                        lamba_l2=args.lambda_l2,
                        min_samples_split=args.min_samples_split,
                        lin_solver=args.lin_solver,
                        new_reg=args.new_reg)

    # Fit training data
    reg.fit_all(X_train, Y_train)
    print("Models learned..")

    # Use the learned model to predict Y for test samples
    reg_predict_dict_test = reg.predict_all(X_test)

    print("Prediction on testing data done..")

    # Use the learned model to predict Y for test samples
    reg_predict_dict_train = reg.predict_all(X_train)

    print("Prediction on training data done..")

    # Classifier list
    reg_list = ["lin_reg", "ransac_reg", "lasso_reg", "ridge_reg", "dt_reg"]
    if args.lin_solver and args.new_reg:
        reg_list.append("lin_reg_solver")
        reg_list.append("neo_reg")
    elif args.lin_solver or args.new_reg:
        if args.lin_solver:
            reg_list.append("lin_reg_solver")
        else:
            reg_list.append("neo_reg")

    # Runtime for building model
    result = pandas.DataFrame(reg.runtime).T
    result.columns = reg_list

    # MSE on test data
    mse_test = []
    # R2 on test data
    r2_test = []
    for reg_m in reg_list:
        Y_pred = reg_predict_dict_test[reg_m]

        # MSE
        mse = mean_squared_error(y_true=Y_test, y_pred=Y_pred)
        # R2
        r2 = r2_score(y_true=Y_test, y_pred=Y_pred)

        mse_test.append(mse)
        r2_test.append(r2)

    mse_test = pandas.DataFrame(mse_test).T
    r2_test = pandas.DataFrame(r2_test).T
    mse_test.columns = reg_list
    r2_test.columns = reg_list
    result = result.append(mse_test, ignore_index=True)
    result = result.append(r2_test, ignore_index=True)

    # MSE on training data
    mse_train = []
    # R2 on training data
    r2_train = []
    for reg_m in reg_list:
        Y_pred = reg_predict_dict_train[reg_m]

        # MSE
        mse = mean_squared_error(y_true=Y_train, y_pred=Y_pred)
        # R2
        r2 = r2_score(y_true=Y_train, y_pred=Y_pred)

        mse_train.append(mse)
        r2_train.append(r2)

    mse_train = pandas.DataFrame(mse_train).T
    r2_train = pandas.DataFrame(r2_train).T
    mse_train.columns = reg_list
    r2_train.columns = reg_list
    result = result.append(mse_train, ignore_index=True)
    result = result.append(r2_train, ignore_index=True)

    result.index = ['runtime', 'mse on test', 'r2 on test', 'mse on training', 'r2 on training']

    return result


def FetchHousePricing(path):

    data = pandas.read_table(path, sep=' ', skipinitialspace=True, header=None)
    header_name = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',' B', 'LSTAT',
                   'MEDV']
    data.columns = header_name
    data = StandardScaler().fit_transform(X=data)
    return data


def FetchCaliforniaRenEnergy(path):

    data = pandas.read_csv(path)
    data = data.drop(columns='TIMESTAMP')  # drop timestamp

    day_count = 1
    month = numpy.zeros(data.shape[0])
    month_count = 1

    solar_pv_stats = numpy.nanmean(data['SOLAR PV'])
    solar_thermal_stats = numpy.nanmean(data['SOLAR THERMAL'])
    solar_stats = [numpy.nanmean(data['SOLAR']), numpy.nanstd(data['SOLAR'])]

    solar_energy = data['SOLAR']
    nan_solar = data['SOLAR'].isnull()

    for i in range(0, data.shape[0]):
        month[i] = month_count
        if data.iloc[i, 3] == 24.0:
            day_count = day_count + 1
        if day_count % 365 == 0:
            month_count = month_count + 1
        if nan_solar[i]:
            solar_energy[i] = (((solar_stats[0]/solar_pv_stats) * data.iloc[i, 6]) +
                               ((solar_stats[0]/solar_thermal_stats) * data.iloc[i, 7]))/2

    data = data.drop(columns=['SOLAR', 'SOLAR PV', 'SOLAR THERMAL'])
    data.insert(0, 'MONTH', month)
    data.insert(7, 'SOLAR', solar_energy)

    solar_data = data[['MONTH', 'Hour', 'SMALL HYDRO', 'SOLAR']]
    wind_data = data[['MONTH', 'SMALL HYDRO', 'SOLAR', 'WIND TOTAL']]
    return solar_data, wind_data


# setup argument names and help options
def setup_arguments():

    arguments = argparse.ArgumentParser()

    arguments.add_argument('-seed', help='int; Random seed. (Default: 1)', default=1, required=False, type=int)
    arguments.add_argument('-n_jobs', help='int; Number of parallel threads to be used. (Default: 4)', default=4,
                           required=False, type=int)
    arguments.add_argument('-max_iter', help='int; Maximum number of epochs for RANSAC, Lasso and Ridge. (Default: 50)',
                           default=50, required=False, type=int)
    arguments.add_argument('-data_to_analyze', help='int; Proj 4 evaluates on 3 datasets, 1) House price, 2) Solar'
                                                    'energy and 3) Wind energy from California Power Production. '
                                                    'if a fourth data set has to be evaluated please set this option to'
                                                    '4 and provide a -data_path, -data_name.[*]', type=int,
                           required=True)
    arguments.add_argument('-data_path', help='Path to the data file; The last column will be treated as response and '
                                              'the rest as attributes.[*]', type=str, required=True)
    arguments.add_argument('-data_name', help='Name of the data-set being evaluated; only valid if -data_to_analyze=4',
                           type=str)

    arguments.add_argument('-min_samples_ransac', help='float; RANSAC : The minimum percentage [0,1] of samples chosen '
                                                       'randomly from original data. (Default: 0.5)', default=0.5,
                           required=False, type=float)

    arguments.add_argument('-lambda_l1', help='float; Lasso : The coefficient [0,1] of L1 norm in Lasso. (Default: 1)',
                           default=1, required=False, type=float)

    arguments.add_argument('-lambda_l2', help='float; Ridge : The coefficient [0,1] of L2 norm in Ridge. (Default: 1)',
                           default=1, required=False, type=float)

    arguments.add_argument('-min_samples_split', help='int; Decision Tree : The minimum number of samples required to '
                                                      'split an internal node. (Default: 25)', default=25,
                           required=False, type=int)
    arguments.add_argument('-lin_solver', help='bool; Should results from linear solver be included? ',
                           const=True, required=False, type=bool, nargs='?')

    arguments.add_argument('-new_reg', help='bool; Should results from new regression method be included? '
                                            '[does not work] ', const=False, required=False, type=bool, nargs='?')
    return arguments

main()