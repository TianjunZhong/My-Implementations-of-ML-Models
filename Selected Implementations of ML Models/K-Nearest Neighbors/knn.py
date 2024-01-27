import argparse

import numpy
import numpy as np
import pandas as pd
from queue import PriorityQueue
import statistics


class Knn(object):
    k = 0  # number of neighbors to use
    train_features = pd.DataFrame()  # placeholder for the training features
    train_labels = pd.Series()   # placeholder for the training labels

    def __init__(self, k):
        """
        Knn constructor

        Parameters
        ----------
        k : int 
            Number of neighbors to use.
        """
        self.k = k

    def train(self, xFeat, y):
        """
        Train the k-nn model.

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """
        # TODO do whatever you need

        # Memorize the data
        features = pd.DataFrame(xFeat)
        labels = pd.Series(y)
        self.train_features = features
        self.train_labels = labels

        return self

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
            Predicted class label per sample
        """
        yHat = []  # variable to store the estimated class label
        # TODO

        # make sure the input is in desired format
        features = pd.DataFrame(xFeat)

        # predict label for all data points
        for i in range(len(features)):
            data_point = features.loc[i]
            yHat.append(self.predict_label(data_point))

        return yHat

    def predict_label(self, data_point: pd.Series):
        '''
        Given a data point in 1-demension, predict its label
        '''
        
        # priority queue (max heap by reversing the priority number) 
        # storing the k nearest neighbors' labels
        neighbor_labels = PriorityQueue(self.k)

        # for each data point, find its k nearest neighbors
        for i in range(len(self.train_features)):
            train_data_point = self.train_features.loc[i]
            distance = euclidean_distance(data_point, train_data_point)

            if not neighbor_labels.full():
                neighbor_labels.put((-distance, self.train_labels[i]))
            else:
                furthest_neighbor = neighbor_labels.get()
                if distance < -furthest_neighbor[0]:
                    neighbor_labels.put((-distance, self.train_labels[i]))
                else:
                    neighbor_labels.put((furthest_neighbor[0], furthest_neighbor[1]))

        # predict the label based on the nearest neighbors
        labels = []
        while not neighbor_labels.empty():
            labels.append(neighbor_labels.get()[1])

        predicted_label = statistics.mode(labels)

        return predicted_label


def euclidean_distance(point1, point2):
    '''
    calculate the Euclidean Distance between to data points
    '''
    point1 = np.array(point1)
    point2 = np.array(point2)

    distance = point1 - point2
    distance = np.dot(distance, distance)
    distance = np.sqrt(distance)

    return distance


def accuracy(yHat, yTrue):
    """
    Calculate the accuracy of the prediction

    Parameters
    ----------
    yHat : 1d-array with shape n
        Predicted class label for n samples
    yTrue : 1d-array with shape n
        True labels associated with the n samples

    Returns
    -------
    acc : float between [0,1]
        The accuracy of the model
    """
    # TODO calculate the accuracy
    acc = 0

    # assure desired format
    yHat = numpy.array(yHat)
    yTrue = numpy.array(yTrue)

    acc = (np.sum(yHat == yTrue)) / yHat.size

    return acc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    parser.add_argument("--xTrain",
                        default="q3xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q3yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q3xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q3yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    # create an instance of the model
    knn = Knn(args.k)
    knn.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = knn.predict(xTrain)
    trainAcc = accuracy(yHatTrain, yTrain['label'])
    # predict the test dataset
    yHatTest = knn.predict(xTest)
    testAcc = accuracy(yHatTest, yTest['label'])
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)


if __name__ == "__main__":
    main()
