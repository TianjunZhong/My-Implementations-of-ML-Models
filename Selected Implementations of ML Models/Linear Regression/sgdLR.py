import argparse
import numpy as np
import pandas as pd
import time

from lr import LinearRegression, file_to_numpy


def grad_pt(beta, xi, yi):
    """
    Calculate the gradient for a mini-batch sample.

    Parameters
    ----------
    beta : 1d array with shape d
    xi : 2d numpy array with shape b x d
        Batch training data
    yi: 2d array with shape bx1
        Array of responses associated with training data.

    Returns
    -------
        grad : 1d array with shape d
    """
    n = xi.shape[0]
    gradient = np.matmul(xi.transpose(), np.matmul(xi, beta) - yi)
    gradient /= n

    return gradient


def shuffle_data(x: np.array, y: np.array):
        rand = np.arange(x.shape[0])
        np.random.shuffle(rand)
        return (x[rand], y[rand])


class SgdLR(LinearRegression):
    lr = 1  # learning rate
    bs = 1  # batch size
    mEpoch = 1000 # maximum epoch size

    def __init__(self, lr, bs, epoch):
        self.lr = lr
        self.bs = bs
        self.mEpoch = epoch

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """
        trainStats = {}
        # TODO: DO SGD
        n = xTrain.shape[0]
        num_batch = int(n / self.bs)
        self.beta = np.ones(xTrain.shape[1]).reshape((xTrain.shape[1], 1))

        iteration_num = 0
        start_time = time.time()
        for epoch in range(self.mEpoch):
            # Randomly shuffle training data and break into B = N/BatchSize batches
            xTrain_shuffled, yTrain_shuffled = shuffle_data(xTrain, yTrain)
            xTrain_split = np.array_split(xTrain_shuffled, num_batch)
            yTrain_split = np.array_split(yTrain_shuffled, num_batch)

            for b in range(num_batch):
                gradient = grad_pt(self.beta, xTrain_split[b], yTrain_split[b])
                self.beta = self.beta - self.lr * gradient

                # calculate time elapsed
                time_elapsed = time.time() - start_time
                # calculate MSE for training and testing
                mse_train = self.mse(xTrain, yTrain)
                mse_test = self.mse(xTest, yTest)

                trainStats.update({
                    iteration_num: {
                        "time": time_elapsed,
                        "train-mse": mse_train,
                        "test-mse": mse_test
                    }
                })
                iteration_num += 1

        return trainStats


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")
    parser.add_argument("lr", type=float, help="learning rate")
    parser.add_argument("bs", type=int, help="batch size")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    # setting the seed for deterministic behavior
    np.random.seed(args.seed)   
    model = SgdLR(args.lr, args.bs, args.epoch)
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)


if __name__ == "__main__":
    main()

