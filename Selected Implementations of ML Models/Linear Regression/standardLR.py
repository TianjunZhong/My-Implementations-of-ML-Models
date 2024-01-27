import argparse
import numpy as np
import pandas as pd
import time

from lr import LinearRegression, file_to_numpy


class StandardLR(LinearRegression):

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """
        trainStats = {}
        # TODO: DO SOMETHING
        start_time = time.time()

        # train the LR model by calculating the beta
        closed_form_first_half = np.linalg.inv((np.matmul(xTrain.transpose(), xTrain)))
        closed_form_second_half = np.matmul(xTrain.transpose(), yTrain)
        self.beta = np.matmul(closed_form_first_half, closed_form_second_half)
        # print(self.beta)

        # calculate time elapsed
        end_time = time.time()
        time_elapsed = end_time - start_time

        # calculate MSE for training and testing
        mse_train = self.mse(xTrain, yTrain)
        mse_test = self.mse(xTest, yTest)

        # update trainStats
        trainStats = {
            0: {
            "time": time_elapsed,
            "train-mse": mse_train,
            "test-mse": mse_test
            }
        }

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

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    model = StandardLR()
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)


if __name__ == "__main__":
    main()
