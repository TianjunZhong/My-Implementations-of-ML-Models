import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import statistics as st


def get_frequency(y: np.array) -> np.array:
    """
    Helper function that find the frequency that each class occurs in a 1d array
    param y: 1d numpy array containing labels
    return: a numpy array containing the frequencies
    """
    # dictionary for counting the frequencies
    label_counts = {}
    for label in y:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1

    frequencies = np.array(list(label_counts.values()))

    return frequencies


def calculate_gini_index(frequency: np.array, size: int):
    """
    Helper function that calculates the gini index.
    param frequency: numpy array containing the frequencies that the classes occur in the set
    param size: size of the data set
    return: gini index of the set
    """
    p = frequency / size
    sum_p_square = np.dot(p, p)
    gini = 1 - sum_p_square

    return gini


def calculate_entropy(frequency: np.array, size: int):
    """
    Helper function that calculates the entropy.
    param frequency: numpy array containing the frequencies that the classes occur in the set
    param size: size of the data set
    return: entropy of the set
    """
    p = frequency / size
    plog2p = p * np.log2(p)
    entropy = -plog2p.sum()

    return entropy


def calculate_split_score(y, criterion):
    """
    Given a numpy array of labels associated with a node, y, 
    calculate the score based on the crieterion specified.

    Parameters
    ----------
    y : numpy.1d array with shape n
        Array of labels associated with a node
    criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
    Returns
    -------
    score : float
        The gini or entropy associated with a node
    """
    frequency = get_frequency(y)

    if criterion == 'gini':
        score = calculate_gini_index(frequency, y.size)
    elif criterion == 'entropy':
        score = calculate_entropy(frequency, y.size)

    return score


class DecisionTree(object):
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion

    def __init__(self, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample
        self.tree = {}

    def train(self, xFeat, y):
        """
        Train the decision tree model.

        Parameters
        ----------
        xFeat : numpy.nd-array with shape n x d
            Training data 
        y : numpy.1d array with shape n
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """
        # TODO do whatever you need
        self.tree = self.decision_tree(xFeat, y, 0)

        return self

    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : numpy.nd-array with shape m x d
            The data to predict.  

        Returns
        -------
        yHat : numpy.1d array with shape m
            Predicted class label per sample
        """
        yHat = np.array([])  # variable to store the estimated class label
        # TODO

        for i in range(xFeat.shape[0]):
            data_point = xFeat[i]
            current_node = self.tree
            
            while True:
                if "predicted_label" in current_node:
                    yHat = np.append(yHat, current_node["predicted_label"])
                    break
                
                data_point_value = data_point[current_node["attr"]]
                if data_point_value <= current_node["val"]:
                    current_node = current_node["left"]
                else:
                    current_node = current_node["right"]

        return yHat

    def decision_tree(self, xFeat, y, depth) -> dict:
        # check stopping criteria
        if depth >= self.maxDepth:
            return {"predicted_label": st.mode(y)}

        # initialize splitting info
        split_attr = None
        split_val = None
        best_gini_or_entropy = 2

        # check each attribute and value for splitting
        for attribute in range(xFeat.shape[1]):
            # sort y by attribute values from xFeat
            order = xFeat[:, attribute].argsort()
            y_sorted = y[order]

            # only consider splitting points that fit the minimun leaf samples criterion
            for value in range(self.minLeafSample, xFeat.shape[0] - self.minLeafSample + 1):
                # partition the y array at the split
                yLeft = y_sorted[0: value]
                yRight = y_sorted[value:]

                # calculate gini index or entropy after splitting
                left_score = calculate_split_score(yLeft, self.criterion)
                right_score = calculate_split_score(yRight, self.criterion)
                score = (yLeft.size / y_sorted.size) * left_score + \
                    (yRight.size / y_sorted.size) * right_score

                # if the score is the new best, update splitting information
                if score < best_gini_or_entropy:
                    split_attr = attribute
                    split_val = value
                    best_gini_or_entropy = score

        # if cannot be splitted, return the majority
        if split_attr == None or split_val == None:
            return {"predicted_label": st.mode(y)}

        # partition the data at the split
        order = xFeat[:, split_attr].argsort()
        xFeat_sorted = xFeat[order]
        xFeatL = xFeat_sorted[0:split_val]
        xFeatR = xFeat_sorted[split_val:]
        y_sorted = y[order]
        yL = y_sorted[0:split_val]
        yR = y_sorted[split_val:]

        # find the numeric value of the split value
        split_val = xFeat_sorted[split_val - 1, split_attr]

        return {"attr": split_attr, 
                "val": split_val, 
                "left": self.decision_tree(xFeatL, yL, depth + 1), 
                "right": self.decision_tree(xFeatR, yR, depth + 1)}


def dt_train_test(dt, xTrain, yTrain, xTest, yTest):
    """
    Given a decision tree model, train the model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    dt : DecisionTree
        The decision tree with the model parameters
    xTrain : numpy.nd-array with shape n x d
        Training data 
    yTrain : numpy.1d array with shape n
        Array of labels associated with training data.
    xTest : numpy.nd-array with shape m x d
        Test data 
    yTest : numpy.1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    acc : float
        The accuracy of the trained knn model on the test data
    """
    # train the model
    dt.train(xTrain, yTrain)
    # predict the training dataset
    yHatTrain = dt.predict(xTrain)
    trainAcc = accuracy_score(yTrain, yHatTrain)
    # predict the test dataset
    yHatTest = dt.predict(xTest)
    testAcc = accuracy_score(yTest, yHatTest)
    return trainAcc, testAcc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("md",
                        type=int,
                        help="maximum depth")
    parser.add_argument("mls",
                        type=int,
                        help="minimum leaf samples")
    parser.add_argument("--xTrain",
                        default="q4xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q4yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q4xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q4yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain).to_numpy()
    yTrain = pd.read_csv(args.yTrain).to_numpy().flatten()
    xTest = pd.read_csv(args.xTest).to_numpy()
    yTest = pd.read_csv(args.yTest).to_numpy().flatten()
    # create an instance of the decision tree using gini
    dt1 = DecisionTree('gini', args.md, args.mls)
    trainAcc1, testAcc1 = dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
    print("GINI Criterion ---------------")
    print("Training Acc:", trainAcc1)
    print("Test Acc:", testAcc1)
    dt = DecisionTree('entropy', args.md, args.mls)
    trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
    print("Entropy Criterion ---------------")
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)


if __name__ == "__main__":
    main()
