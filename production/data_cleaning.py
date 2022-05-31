"""Processors for the data cleaning step of the worklow.

The processors in this step, apply the various cleaning steps identified
during EDA to create the training datasets.
"""
import numpy as np
import os
import tarfile
import urllib.request
from scripts import binned_income_cat
from sklearn.model_selection import StratifiedShuffleSplit

from ta_lib.core.api import (
    custom_train_test_split,
    load_dataset,
    register_processor,
    save_dataset,
)

housing_url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"
HOUSING = os.path.join("../", "data/raw")
housing_path = os.path.join(HOUSING, "housing")
os.makedirs(housing_path, exist_ok=True)
tgz_path = os.path.join(housing_path, "housing.tgz")
urllib.request.urlretrieve(housing_url, tgz_path)
housing_tgz = tarfile.open(tgz_path)
housing_tgz.extractall(path=housing_path)
housing_tgz.close()


@register_processor("data-cleaning", "housing")
def clean_housing_table(context, params):
    """Clean the ``HOUSING`` data table.

    The table contains information on the features of housing price data
    that are used to predict housing price values.
    """

    input_dataset = "raw/housing"
    output_dataset = "cleaned/housing"

    # load dataset
    housing_df = load_dataset(context, input_dataset)

    housing_df_clean = (housing_df
                        # while iterating on testing, it's good to copy the dataset(or a subset)
                        # # as the following steps will mutate the input dataframe. The copy should be
                        # # removed in the production code to avoid introducing perf. bottlenecks.
                        .copy()
                        # set dtypes : nothing to do here
                        .passthrough()
                        .replace({'': np.NaN})
                        # ensure that the key column does not have duplicate records
                        .remove_duplicate_rows(col_names=['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                                                          'total_bedrooms', 'population', 'households', 'median_income',
                                                          'median_house_value', 'ocean_proximity'], keep_first=True)
                        # clean column names (comment out this line while cleaning data above)
                        .clean_names(case_type='snake')
                        )

    # save the dataset
    save_dataset(context, housing_df_clean, output_dataset)

    return housing_df_clean


@register_processor("data-cleaning", "train-test")
def create_training_datasets(context, params):
    """Split the ``HOUSING`` table into ``train`` and ``test`` datasets."""

    input_dataset = "cleaned/housing"
    output_train_features = "train/housing/features"
    output_train_target = "train/housing/target"
    output_test_features = "test/housing/features"
    output_test_target = "test/housing/target"

    # load dataset
    housing_df_processed = load_dataset(context, input_dataset)

    # split the data
    splitter = StratifiedShuffleSplit(n_splits=1,
                                      test_size=params["test_size"],
                                      random_state=context.random_seed)
    housing_df_train, housing_df_test = custom_train_test_split(
        housing_df_processed, splitter, by=binned_income_cat)

    # split train dataset into features and target
    target_col = params["target"]
    train_X, train_y = (
        housing_df_train
        # split the dataset to train and test
        .get_features_targets(target_column_names=target_col)
    )

    # save the train dataset
    save_dataset(context, train_X, output_train_features)
    save_dataset(context, train_y, output_train_target)

    # split test dataset into features and target
    test_X, test_y = (
        housing_df_test
        # split the dataset to train and test
        .get_features_targets(target_column_names=target_col)
    )

    # save the datasets
    save_dataset(context, test_X, output_test_features)
    save_dataset(context, test_y, output_test_target)
