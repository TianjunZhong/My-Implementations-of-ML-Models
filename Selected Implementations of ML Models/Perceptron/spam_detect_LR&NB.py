import numpy as np
from perceptron import file_to_numpy, transform_y, calc_mistakes
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

def main():
    # load the y data
    yTrain = file_to_numpy("yTrain.csv")[:, 1]
    yTest = file_to_numpy("yTest.csv")[:, 1]
    # transform y to -1 and 1
    yTrain = transform_y(yTrain)
    yTest = transform_y(yTest)

    # Naive Bayes
    # the binary dataset
    xTrain = file_to_numpy("xTrain_binary.csv")[:, 1:]
    xTest = file_to_numpy("xTest_binary.csv")[:, 1:]

    nb = MultinomialNB()
    nb.fit(xTrain, yTrain)

    yHat_train = nb.predict(xTrain)
    yHat_test = nb.predict(xTest)

    mistake_train = calc_mistakes(yHat_train, yTrain)
    mistake_test = calc_mistakes(yHat_test, yTest)

    print("For the Naive Bayes model and binary dataset:")
    print(f"{mistake_train} mistakes were made on the training dataset.")
    print(f"{mistake_test} mistakes were made on the testing dataset.")

    print("\n")

    # the count dataset
    xTrain = file_to_numpy("xTrain_count.csv")[:, 1:]
    xTest = file_to_numpy("xTest_count.csv")[:, 1:]

    nb = MultinomialNB()
    nb.fit(xTrain, yTrain)

    yHat_train = nb.predict(xTrain)
    yHat_test = nb.predict(xTest)

    mistake_train = calc_mistakes(yHat_train, yTrain)
    mistake_test = calc_mistakes(yHat_test, yTest)

    print("For the Naive Bayes model and count dataset:")
    print(f"{mistake_train} mistakes were made on the training dataset.")
    print(f"{mistake_test} mistakes were made on the testing dataset.")

    print("\n")

    # Logistic Regression
    # the binary dataset
    xTrain = file_to_numpy("xTrain_binary.csv")[:, 1:]
    dum_feat = np.ones((xTrain.shape[0], 1))
    xTrain = np.append(xTrain, dum_feat, axis=1)

    xTest = file_to_numpy("xTest_binary.csv")[:, 1:]
    dum_feat = np.ones((xTest.shape[0], 1))
    xTest = np.append(xTest, dum_feat, axis=1)

    logreg = LogisticRegression()
    logreg.fit(xTrain, yTrain)

    yHat_train = logreg.predict(xTrain)
    yHat_test = logreg.predict(xTest)

    mistake_train = calc_mistakes(yHat_train, yTrain)
    mistake_test = calc_mistakes(yHat_test, yTest)

    print("For the Logistic Regression model and binary dataset:")
    print(f"{mistake_train} mistakes were made on the training dataset.")
    print(f"{mistake_test} mistakes were made on the testing dataset.")

    print("\n")

    # for the count dataset
    xTrain = file_to_numpy("xTrain_count.csv")[:, 1:]
    dum_feat = np.ones((xTrain.shape[0], 1))
    xTrain = np.append(xTrain, dum_feat, axis=1)

    xTest = file_to_numpy("xTest_count.csv")[:, 1:]
    dum_feat = np.ones((xTest.shape[0], 1))
    xTest = np.append(xTest, dum_feat, axis=1)

    logreg = LogisticRegression()
    logreg.fit(xTrain, yTrain)

    yHat_train = logreg.predict(xTrain)
    yHat_test = logreg.predict(xTest)

    mistake_train = calc_mistakes(yHat_train, yTrain)
    mistake_test = calc_mistakes(yHat_test, yTest)

    print("\n")
    print("For the Logistic Regression model and count dataset:")
    print(f"{mistake_train} mistakes were made on the training dataset.")
    print(f"{mistake_test} mistakes were made on the testing dataset.")


if __name__ == "__main__":
    main()