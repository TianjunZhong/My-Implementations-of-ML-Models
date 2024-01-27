# import numpy as np
import pandas as pd
from sgdLR import SgdLR
from standardLR import StandardLR
from sklearn.model_selection import train_test_split
from lr import file_to_numpy
import matplotlib.pyplot as plt


def main():
    # load the train and test data
    xTrain = file_to_numpy("new_xTrain.csv")
    yTrain = file_to_numpy("eng_yTrain.csv")
    xTest = file_to_numpy("new_xTest.csv")
    yTest = file_to_numpy("eng_yTest.csv")

    lr = 0.001
    epoch = 1
    n = xTrain.shape[0]

    mse_time_list = []
    batch_sizes = [1, int(n/20), int(n/10), int(n/5), int(n/2), n]
    # batch_sizes = [1, 2]

    for i, batch_size in enumerate(batch_sizes):
        time_mse = {"time": [], "train-mse": [], "test-mse": []}

        model = SgdLR(lr, batch_size, epoch + i * 200)
        trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)

        for data in trainStats.values():
            time = data["time"]
            mse_train = data["train-mse"]
            mse_test = data["test-mse"]

            time_mse["time"].append(time)
            time_mse["train-mse"].append(mse_train)
            time_mse["test-mse"].append(mse_test)
    
        mse_time_list.append(time_mse)

    # for i, data in enumerate(mse_time_list):
    #     plt.plot(data["time"], data["train-mse"], label=f"batch size = {batch_sizes[i]}")

    for i, data in enumerate(mse_time_list):
        plt.plot(data["time"], data["test-mse"], label=f"batch size = {batch_sizes[i]}")

    model = StandardLR()
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    # plt.plot(trainStats[0]["time"], trainStats[0]["train-mse"], marker="o", label="closed form")
    plt.plot(trainStats[0]["time"], trainStats[0]["test-mse"], marker="o", label="closed form")

    plt.xlim([0, 2])
    plt.legend()
    # plt.title("Training MSE of Different Batch Sizes")
    plt.title("Testing MSE of Different Batch Sizes")
    plt.xlabel("time")
    plt.ylabel("mse")
    plt.show()

            
            
if __name__ == "__main__":
    main()