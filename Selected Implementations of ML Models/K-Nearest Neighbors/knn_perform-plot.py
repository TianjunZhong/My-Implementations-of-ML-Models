from knn_perform import knn_train_test, standard_scale, minmax_range, add_irr_feature
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    xTrain = pd.read_csv("q4xTrain.csv")
    yTrain = pd.read_csv("q4yTrain.csv")
    xTest = pd.read_csv("q4xTest.csv")
    yTest = pd.read_csv("q4yTest.csv")

    acc_data = {"k": [], "Test Acc (no-preprocessing)": [], "Test Acc (standard scale)": [], 
                "Test Acc (min max scale)": [], "Test Acc (with irrelevant feature)": []}
    
    for k in range(1, 21):
        acc_data["k"].append(k)

        # no preprocessing        
        acc1 = knn_train_test(k, xTrain, yTrain, xTest, yTest)
        acc_data["Test Acc (no-preprocessing)"].append(acc1)

        # preprocess the data using standardization scaling
        xTrainStd, xTestStd = standard_scale(np.array(xTrain), np.array(xTest))
        acc2 = knn_train_test(k, xTrainStd, yTrain, xTestStd, yTest)
        acc_data["Test Acc (standard scale)"].append(acc2)

        # preprocess the data using min max scaling
        xTrainMM, xTestMM = minmax_range(np.array(xTrain), np.array(xTest))
        acc3 = knn_train_test(k, xTrainMM, yTrain, xTestMM, yTest)
        acc_data["Test Acc (min max scale)"].append(acc3)

        # add irrelevant features
        xTrainIrr, yTrainIrr = add_irr_feature(np.array(xTrain), np.array(xTest))
        acc4 = knn_train_test(k, xTrainIrr, yTrain, yTrainIrr, yTest)
        acc_data["Test Acc (with irrelevant feature)"].append(acc4)

    acc_data = pd.DataFrame(acc_data)
    print(acc_data.to_string())

    acc_data.plot.line(x="k", y=["Test Acc (no-preprocessing)", "Test Acc (standard scale)", 
                                 "Test Acc (min max scale)", "Test Acc (with irrelevant feature)"])
    plt.show()
