from knn import Knn, accuracy
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    xTrain = pd.read_csv("q3xTrain.csv")
    yTrain = pd.read_csv("q3yTrain.csv")
    xTest = pd.read_csv("q3xTest.csv")
    yTest = pd.read_csv("q3yTest.csv")

    acc_data = {"k": [], "Train Acc": [], "Test Acc": []}

    for k in range(1, 3):
        # create an instance of the model
        knn = Knn(k)
        knn.train(xTrain, yTrain['label'])
        # predict the training dataset
        yHatTrain = knn.predict(xTrain)
        trainAcc = accuracy(yHatTrain, yTrain['label'])
        # predict the test dataset
        yHatTest = knn.predict(xTest)
        testAcc = accuracy(yHatTest, yTest['label'])

        acc_data["k"].append(k)
        acc_data["Train Acc"].append(trainAcc)
        acc_data["Test Acc"].append(testAcc)

    acc_data = pd.DataFrame(acc_data)
    print(acc_data.to_string())

    acc_data.plot.line(x="k", y=["Train Acc", "Test Acc"])
    plt.show()


