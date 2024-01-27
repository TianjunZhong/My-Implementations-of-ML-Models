import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from itertools import chain
import json


def model_assessment(filename):
    """
    Given the entire data, split it into training and test set 
    so you can assess your different models 
    to compare perceptron, logistic regression,
    and naive bayes. 
    """
    data = pd.DataFrame(columns=["y", "text"])

    # read in the data
    with open(filename, "r") as file:
        while (True):
            line = file.readline()
            if not line:
                break

            line = line.split(" ", 1)
            line[1] = line[1].strip().split(" ")

            data.loc[len(data)] = line

    # split the dataset into training (80%) and testing (20%) data
    testSize = 0.2
    textTrain, textTest, yTrain, yTest = train_test_split(data["text"], data["y"], test_size=testSize)

    train = pd.concat([yTrain, textTrain], axis=1)
    test = pd.concat([yTest, textTest], axis=1)

    return train, test


def build_vocab_map(traindf):
    """
    Construct the vocabulary map such that it returns
    (1) the vocabulary dictionary contains words as keys and
    the number of emails the word appears in as values, and
    (2) a list of words that appear in at least 30 emails.

    ---input:
    dataset: pandas dataframe containing the 'text' column
             and 'y' label column

    ---output:
    dict: key-value is word-count pair
    list: list of words that appear in at least 30 emails
    """
    text_list = traindf["text"]
    word_set = {word for text in text_list for word in text}

    vocab_map = {key: 0 for key in word_set}

    i = 1
    print("vocab map started")

    for text in text_list:

        if i % 100 == 0:
            print(f"email #{i}")
        i += 1

        for word in vocab_map:
            if word in text:
                vocab_map[word] += 1

    print(f"vocab map completed with {len(vocab_map)} words")

    i = 1
    print("word list started")

    word_list = []
    for word, count in vocab_map.items():

        if i % 100 == 0:
            print(f"word #{i}")
        i += 1

        if count >= 30:
            word_list.append(word)

    print(f"word list completed with {len(word_list)} words")

    return vocab_map, word_list


def construct_binary(dataset, freq_words):
    """
    Construct email datasets based on
    the binary representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is 1 if the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise

    ---input:
    dataset: pandas dataframe containing the 'text' column

    freq_word: the vocabulary map built in build_vocab_map()

    ---output:
    numpy array
    """
    feat_vector_list = []

    i = 1
    print("binary construcitons started")

    for email in dataset["text"]:

        if i % 100 == 0:
            print(f"binary: email #{i}")
        i += 1

        feat_vector = []

        for word in freq_words:
            if word in email:
                feat_vector.append(1)
            else:
                feat_vector.append(0)

        feat_vector_list.append(feat_vector)

    feat_vector_list = np.array(feat_vector_list)

    print("binary construction completed")

    return feat_vector_list


def construct_count(dataset, freq_words):
    """
    Construct email datasets based on
    the count representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is the number of times the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise

    ---input:
    dataset: pandas dataframe containing the 'text' column

    freq_word: the vocabulary map built in build_vocab_map()

    ---output:
    numpy array
    """
    feat_vector_list = []

    i = 1
    print("count construcitons started")

    for email in dataset["text"]:

        if i % 100 == 0:
            print(f"count: email #{i}")
        i += 1

        feat_vector = []

        for word in freq_words:
            feat_vector.append(email.count(word))

        feat_vector_list.append(feat_vector)

    feat_vector_list = np.array(feat_vector_list)

    print("count construction completed")

    return feat_vector_list


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default="spamAssassin.data",
                        help="filename of the input data")
    args = parser.parse_args()
    
    train, test = model_assessment(args.data)
    vocab_map, word_list = build_vocab_map(train)

    with open("vocabulary_list.json", "w") as file:
        json.dump(word_list, file)

    yTrain = train["y"]
    yTrain.to_csv("yTrain.csv")

    yTest = test["y"]
    yTest.to_csv("yTest.csv")

    xTrain_binary = construct_binary(train, word_list)
    xTrain_binary = pd.DataFrame(xTrain_binary)
    xTrain_binary.to_csv("xTrain_binary.csv")

    xTest_binary = construct_binary(test, word_list)
    xTest_binary = pd.DataFrame(xTest_binary)
    xTest_binary.to_csv("xTest_binary.csv")

    xTrain_count = construct_count(train, word_list)
    xTrain_count = pd.DataFrame(xTrain_count)
    xTrain_count.to_csv("xTrain_count.csv")

    xTest_count = construct_count(test, word_list)
    xTest_count = pd.DataFrame(xTest_count)
    xTest_count.to_csv("xTest_count.csv")



if __name__ == "__main__":
    main()
