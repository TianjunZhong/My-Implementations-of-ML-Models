# import numpy as np
import pandas as pd
from sgdLR import SgdLR
from sklearn.model_selection import train_test_split
from lr import file_to_numpy
import matplotlib.pyplot as plt


def main():
    # load the train and test data
    xTrain = file_to_numpy("new_xTrain.csv")
    yTrain = file_to_numpy("eng_yTrain.csv")
    xTest = file_to_numpy("new_xTest.csv")
    yTest = file_to_numpy("eng_yTest.csv")

    
    # max_epoch = 11
    # batch_size = 1
    # learning_rate = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    # mse_dict = {}
    # mse_dict["epoch"] = list(range(1, max_epoch))

    # for lr in learning_rate:
        
    #     mse_list = []
    #     for epoch in range(1, max_epoch):

    #         model = SgdLR(lr, batch_size, epoch)
    #         xTrain_40, x_removed, yTrain_40, y_removed = train_test_split(xTrain, yTrain, test_size=0.6)
    #         trainStats = model.train_predict(xTrain_40, yTrain_40, xTest, yTest)

    #         last_key = max(trainStats.keys())
    #         mse_train = trainStats[last_key]["train-mse"]
    #         mse_list.append(mse_train)

    #     mse_dict[f"train-mse_{lr}"] = mse_list

    # mse_data = pd.DataFrame(mse_dict)
    # print(mse_data)

    # mse_data.plot.line(x="epoch", y = [
    #     f"train-mse_{learning_rate[0]}", 
    #     f"train-mse_{learning_rate[1]}", 
    #     f"train-mse_{learning_rate[2]}", 
    #     f"train-mse_{learning_rate[3]}", 
    #     f"train-mse_{learning_rate[4]}"
    # ])
    # plt.show()


    max_epoch = 21
    batch_size = 1
    learning_rate = 0.001

    mse_train_list = []
    mse_test_list = []
    for epoch in range(1, max_epoch):

        model = SgdLR(learning_rate, batch_size, epoch)
        trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)

        last_key = max(trainStats.keys())
        mse_train = trainStats[last_key]["train-mse"]
        mse_train_list.append(mse_train)
        mse_test = trainStats[last_key]["test-mse"]
        mse_test_list.append(mse_test)

    mse_dict = {
        "epoch": list(range(1, max_epoch)), 
        f"train-mse_{learning_rate}": mse_train_list,
        f"test-mse_{learning_rate}": mse_test_list
    }

    mse_data = pd.DataFrame(mse_dict)
    mse_data.plot.line(x="epoch", y = [
        f"train-mse_{learning_rate}", 
        f"test-mse_{learning_rate}"
    ])
    plt.show()




if __name__ == "__main__":
    main()

