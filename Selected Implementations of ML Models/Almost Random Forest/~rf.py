import argparse
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def generate_bootstrap(xTrain, yTrain):
    """
    Helper function to generate a bootstrap sample from the data. Each
    call should generate a different random bootstrap sample!

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data 
    yTrain : 1d array with shape n
        Array of responses associated with training data.

    Returns
    -------
    xBoot : nd-array with shape n x d
        Bootstrap sample from xTrain
    yBoot : 1d array with shape n
        Array of responses associated with xBoot
    oobIdx : 1d array with shape k (which can be 0-(n-1))
        Array containing the out-of-bag sample indices from xTrain 
        such that using this array on xTrain will yield a matrix 
        with only the out-of-bag samples (i.e., xTrain[oobIdx, :]).
    """
    bootIdx = np.random.randint(0, xTrain.shape[0], xTrain.shape[0])
    xBoot = xTrain[bootIdx, :]
    yBoot = yTrain[bootIdx]

    allIdx = np.arange(0, xTrain.shape[0])
    oobIdx = np.setdiff1d(allIdx, bootIdx)

    return xBoot, yBoot, oobIdx


def generate_subfeat(xTrain, maxFeat):
    """
    Helper function to generate a subset of the features from the data. Each
    call is likely to yield different columns (assuming maxFeat is less than
    the original dimension)

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data 
    maxFeat : int
        Maximum number of features to consider in each tree

    Returns
    -------
    xSubfeat : nd-array with shape n x maxFeat
        Subsampled features from xTrain
    featIdx: 1d array with shape maxFeat
        Array containing the subsample indices of features from xTrain
    """
    featIdx = np.random.choice(xTrain.shape[1], maxFeat, replace=False)
    xSubfeat = xTrain[:, featIdx]

    return xSubfeat, featIdx


class RandomForest(object):
    nest = 0           # number of trees
    maxFeat = 0        # maximum number of features
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion
    model = {}         # keeping track of all the models developed, where
                       # the key is the bootstrap sample. The value should be a dictionary
                       # and have 2 keys: "tree" to store the tree built
                       # "feat" to store the corresponding featIdx used in the tree


    def __init__(self, nest, maxFeat, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        nest: int
            Number of trees to have in the forest
        maxFeat: int
            Maximum number of features to consider in each tree
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.nest = nest
        self.maxFeat = maxFeat
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample

    def train(self, xFeat, y):
        """
        Train the random forest using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the number of trees and
            the values are the out of bag errors
        """
        stats = {}
        oobPredictions = {}
        for b in range(self.nest):
            # choose random subspace
            xBoot, yBoot, oobIdx = generate_bootstrap(xFeat, y)
            xSubfeat, featIdx = generate_subfeat(xBoot, self.maxFeat)

            # build individual decision tree
            dt = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.maxDepth, min_samples_leaf=self.minLeafSample)
            dt.fit(xSubfeat, yBoot)

            # update the model
            self.model[len(self.model)] = {"tree": dt, "feat": featIdx}

            # OOB error
            xOOB = xFeat[oobIdx]
            xOOB = xOOB[:, featIdx]

            yHatOOB = dt.predict(xOOB)
            for idx, prediction in zip(oobIdx, yHatOOB):
                if idx in oobPredictions:
                    oobPredictions[idx].append(prediction)
                else:
                    oobPredictions[idx] = [prediction]

            numErr = 0
            for idx, predictions in oobPredictions.items():
                majority_vote = np.bincount(predictions).argmax()
                if majority_vote != y[idx]:
                    numErr += 1
            oobErr = numErr / len(oobPredictions)

            stats[b + 1] = oobErr

        return stats

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
        yHat = []
        votes = []
        for tree_feat in self.model.values():
            tree = tree_feat["tree"]
            feat = tree_feat["feat"]
            xSubFeat = xFeat[:, feat]
            predictions = tree.predict(xSubFeat)
            votes.append(predictions)

        votes = np.array(votes).T
        for vote_list in votes:
            majority_vote = np.bincount(vote_list).argmax()
            yHat.append(majority_vote)

        return yHat


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


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
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")
    
    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    np.random.seed(args.seed)   
    model = RandomForest(args.epoch)
    trainStats = model.train(xTrain, yTrain)
    print(trainStats)
    yHat = model.predict(xTest)


if __name__ == "__main__":
    main()