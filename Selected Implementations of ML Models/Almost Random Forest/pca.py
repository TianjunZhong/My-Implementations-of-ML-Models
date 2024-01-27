import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def normalize_feat(xTrain, xTest):
    """
    Preprocess the training data to have zero mean and unit variance.
    """
    transform_standard(xTrain)
    transform_standard(xTest)

    return xTrain, xTest


def transform_standard(data: np.array):
    '''
    transform a numpy array with mean 0 and unit variance
    '''
    for i in range(data.shape[1]):
        feature = data[:, i]
        mean = np.mean(feature)
        std = np.std(feature)
        feature = (feature - mean) / std
        data[:, i] = feature


def unreg_log(xTrain, yTrain, xTest, yTest):
    '''
    Train an unregularized logistic regression model on the 
    dataset and predict the probabilities on the test data 
    and calculate the ROC.
    '''
    logreg = LogisticRegression(penalty=None, max_iter=100)
    logreg.fit(xTrain, yTrain)

    # probabilty estimates
    yHatTestProb = logreg.predict_proba(xTest)
    # FPR, TPR, AUC
    fpr, tpr, thresholds = metrics.roc_curve(yTest, yHatTestProb[:, 1])
    auc = metrics.auc(fpr, tpr)

    return fpr, tpr, auc


def run_pca(xTrain, xTest):
    n = find_n_components(xTrain)
    pca = PCA(n_components=n)
    xTrain_new = pca.fit_transform(xTrain)
    xTest_new = pca.transform(xTest)

    return xTrain_new, xTest_new, pca.components_
    

def find_n_components(xTrain):
    d = xTrain.shape[1]
    pca = PCA(n_components=d)
    pca.fit(xTrain)

    exp_var = pca.explained_variance_ratio_
    exp_var_sum = 0
    for n, ratio in enumerate(exp_var):
        exp_var_sum += ratio
        if exp_var_sum >= 0.95:
            return n + 1


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


def main():
    xTrain = file_to_numpy("xTrain.csv")
    xTest = file_to_numpy("xTest.csv")
    yTrain = file_to_numpy("yTrain.csv").ravel()
    yTest = file_to_numpy("yTest.csv").ravel()

    xTrain, xTest = normalize_feat(xTrain, xTest)
    fpr, tpr, auc = unreg_log(xTrain, yTrain, xTest, yTest)

    xTrain_new, xTest_new, components = run_pca(xTrain, xTest)
    fpr_pca, tpr_pca, auc_pca = unreg_log(xTrain_new, yTrain, xTest_new, yTest)

    plt.plot(fpr, tpr, label="normalized")
    plt.plot(fpr_pca, tpr_pca, label="pca")
    plt.legend()
    plt.title("ROC Curver")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()






if __name__ == "__main__":
    main()
    