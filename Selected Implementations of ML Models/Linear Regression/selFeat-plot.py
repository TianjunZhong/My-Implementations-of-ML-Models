import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt 
from selFeat import extract_features, cal_corr


def main():
    # load the train data
    xTrain = pd.read_csv("eng_xTrain.csv")
    yTrain = pd.read_csv("eng_yTrain.csv")
    # extract the new features
    xNewTrain = extract_features(xTrain)
    # append the labels to new training data
    xNewTrain["label"] = yTrain
    # calculate the correlation matrix
    corrMat = cal_corr(xNewTrain)
    print(corrMat.loc["label"])
    # plot the heatmap
    sn.heatmap(data=corrMat, xticklabels=True, yticklabels=True)
    plt.show()


if __name__ == "__main__":
    main()