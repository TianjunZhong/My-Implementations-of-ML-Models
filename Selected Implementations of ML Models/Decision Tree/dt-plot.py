import pandas as pd
from dt import DecisionTree, dt_train_test
import matplotlib.pyplot as plt


def main():
    # load the train and test data
    xTrain = pd.read_csv("q4xTrain.csv").to_numpy()
    yTrain = pd.read_csv("q4yTrain.csv").to_numpy().flatten()
    xTest = pd.read_csv("q4xTest.csv").to_numpy()
    yTest = pd.read_csv("q4yTest.csv").to_numpy().flatten()

    # initialize dictionary to store accuracy info
    acc_md = {"max depth": [], "train acc": [], "test acc": []}
    acc_mls = {"min leaf sample": [], "train acc": [], "test acc": []}

    for i in range(50):
        # create a decision tree with varying max depth and fixed min leaf sample
        dt_md = DecisionTree("entropy", i, 1)
        # find the accuracies
        train_acc, test_acc = dt_train_test(
            dt_md, xTrain, yTrain, xTest, yTest)
        # update the data
        acc_md["max depth"].append(i)
        acc_md["train acc"].append(train_acc)
        acc_md["test acc"].append(test_acc)

        # create a decision tree with fixed max depth and varying min leaf sample
        dt_mls = DecisionTree("entropy", 100, i + 1)
        # find the accuracies
        train_acc, test_acc = dt_train_test(
            dt_mls, xTrain, yTrain, xTest, yTest)
        # update the data
        acc_mls["min leaf sample"].append(i + 1)
        acc_mls["train acc"].append(train_acc)
        acc_mls["test acc"].append(test_acc)

    # plots
    acc_md = pd.DataFrame(acc_md)
    acc_md.plot.line(x="max depth", y=["train acc", "test acc"])
    acc_mls = pd.DataFrame(acc_mls)
    acc_mls.plot.line(x="min leaf sample", y=["train acc", "test acc"])
    plt.show()


if __name__ == "__main__":
    main()
