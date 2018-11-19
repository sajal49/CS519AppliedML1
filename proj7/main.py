from sklearn.datasets import fetch_mldata
import numpy
from ConvNN import ConvNN
from sklearn.model_selection import train_test_split
import argparse


def main():
    # setup args list
    arguments = setup_arguments()
    args = arguments.parse_args()

    mnist = fetch_mldata('MNIST original')
    X, Y = mnist['data'], mnist['target']

    means = numpy.mean(X, axis=0)
    std_dev = numpy.std(X)
    X = (X - means)/std_dev

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.85, stratify=Y)
    X_train, X_validate, Y_train, Y_validate = train_test_split(X_train, Y_train, train_size=0.85, stratify=Y_train)

    cnn = ConvNN(batch_size=args.batch_size,
                 epochs=args.epochs,
                 learning_rate=args.learning_rate,
                 random_seed=args.seed,
                 c1_kernel_size=args.c1_kernel_size,
                 p1_pool_size=args.p1_pool_size,
                 c2_kernel_size=args.c2_kernel_size,
                 p2_pool_size=args.p2_pool_size,
                 c1_op_channel=args.c1_op_channel,
                 c2_op_channel=args.c2_op_channel)

    cnn.train(X_train=X_train, Y_train=Y_train, X_validate=X_validate, Y_validate=Y_validate)
    cnn.save(epoch=args.epochs)
    preds = cnn.predict(X_test, type='labels')
    print(numpy.sum(Y_test == preds)/len(Y_test) + "% accuracy achieved on test data")


# setup argument names and help options
def setup_arguments():
    arguments = argparse.ArgumentParser()

    arguments.add_argument('-seed', help='int; Random seed. (Default: 1)', default=1, required=False, type=int)
    arguments.add_argument('-batch_size', help='int; Batch size for CNN (Default: 100)', default=100,
                           required=False, type=int)
    arguments.add_argument('-epochs', help='int; Number of iterations / epochs to consider (Default: 50)',
                           type=int, default=50, required=False)
    arguments.add_argument('-learning_rate', help='float; Learning rate (Default: 1e-4).', type=float, default=1e-4,
                           required=False)
    arguments.add_argument('-c1_kernel_size', help='int; Kernel size for 1st conv layer (Default: 3).', type=int,
                           default=3, required=False)
    arguments.add_argument('-c2_kernel_size', help='int; Kernel size for 2nd conv layer (Default: 3).', type=int,
                           default=3, required=False)
    arguments.add_argument('-c1_op_channel', help='int; Number of open channels for 1st conv layer (Default: 4).',
                           type=int, default=4, required=False)
    arguments.add_argument('-c2_op_channel', help='int; Number of open channels for 2nd conv layer (Default: 2).',
                           type=int, default=2, required=False)
    arguments.add_argument('-p1_pool_size', help='int; Pool size for 1st pool layer (Default: 2).', type=int,
                           default=2, required=False)
    arguments.add_argument('-p2_pool_size', help='int; Pool size for 2nd pool layer (Default: 4).', type=int,
                           default=4, required=False)
    return arguments


main()