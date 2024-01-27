import argparse
import numpy as np
import pandas as pd
from datetime import datetime


def extract_features(df):
    """
    Given a pandas dataframe, extract the relevant features
    from the date column

    Parameters
    ----------
    df : pandas dataframe
        Training or test data 
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with the new features
    """
    # TODO do more than this
    # Extract time in the day from dates as number of minutes since 12am
    time_list = []
    for date_time in df["date"]:
        date_time = datetime.strptime(date_time, '%m/%d/%y %H:%M')
        seconds_since_midnight = (date_time - date_time.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
        minutes_since_midnight = int(seconds_since_midnight / 60)
        time_list.append(minutes_since_midnight)
    df["time"] = pd.Series(time_list)

    # Convert dates to weekdays
    weekdays = []
    for date_time in df["date"]:
        weekday = datetime.strptime(date_time, '%m/%d/%y %H:%M').weekday()
        weekdays.append(weekday)
    df["weekday"] = pd.Series(weekdays)

    df = df.drop(columns=['date'])
    return df


def cal_corr(df):
    """
    Given a pandas dataframe (include the target variable at the last column), 
    calculate the correlation matrix (compute pairwise correlation of columns)

    Parameters
    ----------
    df : pandas dataframe
        Training or test data (with target variable)
    Returns
    -------
    corrMat : pandas dataframe
        Correlation matrix
    """
    # TODO
    # calculate the correlation matrix and perform the heatmap
    corrMat = df.corr(method="pearson")
    return corrMat


def select_features(df):
    """
    Select the features to keep

    Parameters
    ----------
    df : pandas dataframe
        Training or test data 
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with a subset of the columns
    """
    # TODO
    df = df[["lights", "T2", "RH_out", "time"]]
    return df


def transform_standard(data: pd.DataFrame):
    '''
    transform the training and test data to have 0 mean and unit variance
    '''
    # Convert dataframe to numpy array which is faster and easier to use
    data_np = data.to_numpy()

    # standardize the data values
    for i in range(data_np.shape[1]):
        feature = data_np[:, i]
        mean = np.mean(feature)
        std = np.std(feature)
        feature = (feature - mean) / std
        data_np[:, i] = feature

    # Convert the numpy array back to dataframe and return
    data_pd = pd.DataFrame(data_np)
    data_pd.columns = ["lights", "T2", "RH_out", "time"]
    return data_pd


def preprocess_data(trainDF, testDF):
    """
    Preprocess the training data and testing data

    Parameters
    ----------
    trainDF : pandas dataframe
        Training data 
    testDF : pandas dataframe
        Test data 
    Returns
    -------
    trainDF : pandas dataframe
        The preprocessed training data
    testDF : pandas dataframe
        The preprocessed testing data
    """
    # TODO do something
    trainDF = transform_standard(trainDF)
    testDF = transform_standard(testDF)

    return trainDF, testDF


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("outTrain",
                        help="filename of the updated training data")
    parser.add_argument("outTest",
                        help="filename of the updated test data")
    parser.add_argument("--trainFile",
                        default="eng_xTrain.csv",
                        help="filename of the training data")
    parser.add_argument("--testFile",
                        default="eng_xTest.csv",
                        help="filename of the test data")
    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.trainFile)
    xTest = pd.read_csv(args.testFile)
    # extract the new features
    xNewTrain = extract_features(xTrain)
    xNewTest = extract_features(xTest)
    # select the features
    xNewTrain = select_features(xNewTrain)
    xNewTest = select_features(xNewTest)
    # preprocess the data
    xTrainTr, xTestTr = preprocess_data(xNewTrain, xNewTest)
    # save it to csv
    xTrainTr.to_csv(args.outTrain, index=False)
    xTestTr.to_csv(args.outTest, index=False)


if __name__ == "__main__":
    main()
