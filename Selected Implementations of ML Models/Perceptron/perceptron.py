import argparse
import numpy as np
import pandas as pd
import time
import csv

class Perceptron(object):
    mEpoch = 1000  # maximum epoch size
    w = None       # weights of the perceptron

    def __init__(self, epoch):
        self.mEpoch = epoch

    def train(self, xFeat, y):
        """
        Train the perceptron using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the epochs and values the number of mistakes
        """
        stats = {}
        # TODO implement this
        for epoch in range(1, self.mEpoch + 1):
            num_mistake = 0

            for row in range(xFeat.shape[0]):
                wnew, mistake = self.sample_update(xFeat[row], y[row])
                self.w = wnew
                num_mistake += mistake

            stats[epoch] = num_mistake

            if num_mistake == 0:
                break

            # shuffle the data
            rand = np.arange(xFeat.shape[0])
            np.random.shuffle(rand)
            xFeat = xFeat[rand]
            y = y[rand]

        return stats

    def sample_update(self, xi, yi):
        """
        Given a single sample, give the resulting update to the weights

        Parameters
        ----------
        xi : numpy array of shape 1 x d
            Training sample 
        y : single value (-1, +1)
            Training label

        Returns
        -------
            wnew: numpy 1d array
                Updated weight value
            mistake: 0/1 
                Was there a mistake made 
        """
        # print("The following is xi used for the sample_update function:")
        # print(f"shape of xi: {xi.shape}")
        # print("xi data:")
        # print(xi)
        if len(xi.shape) == 1:
            xi = np.array([xi])

        yHat = self.predict(xi)[0]

        if yHat == yi:
            wnew = self.w
            mistake = 0
        else:
            wnew = self.w + xi[0] * yi
            mistake = 1

        return wnew, mistake

    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.  

        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted response per sample
        """
        if self.w is None:
            self.w = np.random.rand(xFeat.shape[1])

        yHat = np.dot(xFeat, self.w)

        yHat[yHat >= 0] = 1
        yHat[yHat < 0] = -1

        return yHat


def transform_y(y):
    """
    Given a numpy 1D array with 0 and 1, transform the y 
    label to be -1 and 1

    Parameters
    ----------
    y : numpy 1-d array with labels of 0 and 1
        The true label.      

    Returns
    -------
    y : numpy 1-d array with labels of -1 and 1
        The true label but 0->-1
    """
    y[y == 0] = -1
    return y

def calc_mistakes(yHat, yTrue):
    """
    Calculate the number of mistakes
    that the algorithm makes based on the prediction.

    Parameters
    ----------
    yHat : 1-d array or list with shape n
        The predicted label.
    yTrue : 1-d array or list with shape n
        The true label.      

    Returns
    -------
    err : int
        The number of mistakes that are made
    """
    num = np.sum(yHat != yTrue)
    return num


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()



def tune_perceptron(trainx, trainy, epochList):
    """
    Tune the preceptron to find the optimal number of epochs

    Parameters
    ----------
    trainx : a nxd numpy array
        The input from either binary / count matrix
    trainy : numpy 1d array of shape n
        The true label.    
    epochList: a list of positive integers
        The epoch list to search over  

    Returns
    -------
    epoch : int
        The optimal number of epochs
    """
    k = 5
    best_mistake = trainx.shape[0]
    best_epoch = 0

    # split the data into k chunks
    x_split = np.array_split(trainx, k)
    y_split = np.array_split(trainy, k)

    for epoch in epochList:
        model = Perceptron(epoch)
        test_mistake = 0

        # k-fold cross validation
        for i in range(k):
            # merge chunks to get training and testing data
            xTrain = x_split[0:i] + x_split[i+1:]
            xTrain = np.array([row for chunk in xTrain for row in chunk])
            xTest = x_split[i]
            yTrain = y_split[0:i] + y_split[i+1:]
            yTrain = np.array([row for chunk in yTrain for row in chunk])
            yTest = y_split[i]

            model.train(xTrain, yTrain)
            yHat = model.predict(xTest)
            test_mistake += calc_mistakes(yHat, yTest)

        if test_mistake < best_mistake:
            best_mistake = test_mistake
            best_epoch = epoch
        
    return best_epoch


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrain",
                        default="xTrain_binary.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="xTest_binary.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="yTest.csv",
                        help="filename for labels associated with the test data")
    parser.add_argument("--epoch", type=int, default=1000, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")
    
    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)[:, 1:]
    dum_feat = np.ones((xTrain.shape[0], 1))
    xTrain = np.append(xTrain, dum_feat, axis=1)
    yTrain = file_to_numpy(args.yTrain)[:, 1]
    xTest = file_to_numpy(args.xTest)[:, 1:]
    dum_feat = np.ones((xTest.shape[0], 1))
    xTest = np.append(xTest, dum_feat, axis=1)
    yTest = file_to_numpy(args.yTest)[:, 1]
    # transform to -1 and 1
    yTrain = transform_y(yTrain)
    yTest = transform_y(yTest)

    np.random.seed(args.seed)   
    model = Perceptron(args.epoch)
    trainStats = model.train(xTrain, yTrain)
    print(trainStats)
    yHat = model.predict(xTest)
    # print out the number of mistakes
    print("Number of mistakes on the test dataset")
    print(calc_mistakes(yHat, yTest))

    # print(tune_perceptron(xTrain, yTrain, [1, 10] + list(range(100, 101, 100))))


if __name__ == "__main__":
    main()