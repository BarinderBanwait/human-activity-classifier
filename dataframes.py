from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
from sklearn.model_selection import train_test_split
import re

# Declare location of data files

X_TRAIN_LOCATION = "UCI_HAR_Dataset/train/X_train.txt"
Y_TRAIN_LOCATION = "UCI_HAR_Dataset/train/y_train.txt"

X_TEST_LOCATION = "UCI_HAR_Dataset/test/X_test.txt"
Y_TEST_LOCATION = "UCI_HAR_Dataset/test/y_test.txt"

COLUMN_NAMES_LOCATION = "UCI_HAR_Dataset/features.txt"

ACTIVITY_LABELS_LOCATION = "UCI_HAR_Dataset/activity_labels.txt"

# Load the data as pandas dataframes

X_train = pd.read_csv(X_TRAIN_LOCATION, header=None, delim_whitespace=True)
Y_train = pd.read_csv(Y_TRAIN_LOCATION, header=None, delim_whitespace=True)

X_test = pd.read_csv(X_TEST_LOCATION, header=None, delim_whitespace=True)
Y_test = pd.read_csv(Y_TEST_LOCATION, header=None, delim_whitespace=True)

column_names = pd.read_csv(COLUMN_NAMES_LOCATION, header=None, delim_whitespace=True)

def clean_strings(line):
  line = re.sub('[(),]', '', line)
  return str(line)

column_names_clean = column_names[1].apply(clean_strings)

activity_labels_df = pd.read_csv(ACTIVITY_LABELS_LOCATION, header=None, names = ['key', 'meaning'], delim_whitespace=True)
activity_labels_dicty = pd.Series(activity_labels_df.meaning.values, index=activity_labels_df.key).to_dict()
NO_OF_CLASSES = len(activity_labels_dicty)

activity_labels_dict = {i:activity_labels_dicty[i+1] for i in range(NO_OF_CLASSES)}  # zero-indexing

# Rename columns on the 'X' dataframes

X_train.columns = column_names_clean
X_test.columns = column_names_clean
del X_train.columns.name
del X_test.columns.name
# Combine X and Y dataframes for the tensorflow pipeline

def get_dataframes():

    train = X_train.copy()
    train['target'] = Y_train - 1

    test = X_test.copy()
    test['target'] = Y_test - 1

    # Create a validation set

    train, val = train_test_split(train, test_size=0.2)

    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    return train, val, test

train, val, test = get_dataframes()