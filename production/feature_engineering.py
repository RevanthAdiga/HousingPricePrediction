"""Processors for the feature engineering step of the worklow.

The step loads cleaned training data, processes the data for outliers,
missing values and any other cleaning steps based on strategic rules/intuition.

The trained pipeline and any artifacts are then saved to be used in
training/scoring pipelines.
"""
import logging
import numpy as np
import os.path as op
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ta_lib.core.api import (
    DEFAULT_ARTIFACTS_PATH,
    get_dataframe,
    get_feature_names_from_column_transformer,
    load_dataset,
    register_processor,
    save_pipeline,
)
from ta_lib.data_processing.api import Outlier

logger = logging.getLogger(__name__)

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


@register_processor("feat-engg", "transform-features")
def transform_features(context, params):
    """Transform dataset to create training datasets."""

    input_features_ds = "train/housing/features"
    input_target_ds = "train/housing/target"

    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    # load datasets
    train_X = load_dataset(context, input_features_ds)
    train_y = load_dataset(context, input_target_ds)

    # Treating Outliers
    outlier_transformer = Outlier(method=params["outliers"]["method"])
    train_X = outlier_transformer.fit_transform(
        train_X, drop=params["outliers"]["drop"])

    attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=True)
    housing_extra_attribs = attr_adder.transform(train_X.values)

    col_names = "total_rooms", "total_bedrooms", "population", "households"
    rooms_ix, bedrooms_ix, population_ix, households_ix = [
        train_X.columns.get_loc(c) for c in col_names
    ]

    train_X = pd.DataFrame(housing_extra_attribs,
                           columns=list(train_X.columns) + [
                               "rooms_per_household",
                               "population_per_household", "bedrooms_per_room"
                           ],
                           index=train_X.index)

    train_X_num = train_X.drop("ocean_proximity", axis=1)

    # NOTE: You can use ``Pipeline`` to compose a collection of transformers
    # into a single transformer. In this case, we are composing a
    # ``StandardScalar`` and a ``SimpleImputer`` to first encode the
    # variable into a scaled numerical values and then impute any missing
    # values using ``most_frequent`` strategy.

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    num_attribs = list(train_X_num)

    cat_attribs = ["ocean_proximity"]

    # NOTE: the list of transformations here are not sequential but weighted
    # (if multiple transforms are specified for a particular column)

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

    # Check if the data should be sampled. This could be useful to quickly run
    # the pipeline for testing/debugging purposes (undersample)
    # or profiling purposes (oversample).
    # The below is an example how the sampling can be done on the train data if required.
    # Model Training in this reference code has been done on complete train data itself.

    sample_X = train_X.sample(frac=0.1, random_state=context.random_seed)
    sample_y = train_y.loc[sample_X.index]
    sample_train_X = get_dataframe(
        full_pipeline.fit_transform(sample_X, sample_y),
        get_feature_names_from_column_transformer(full_pipeline),
    )
    # nothing to do for target
    sample_train_y = sample_y

    print(sample_train_X, sample_train_y)

    # Train the feature engg. pipeline prepared earlier. Note that the pipeline is
    # fitted on only the **training data** and not the full dataset.
    # This avoids leaking information about the test dataset when training the model.
    # In the below code train_X, train_y in the fit_transform can be replaced with
    # sample_X and sample_y if required.
    train_X = get_dataframe(
        full_pipeline.fit_transform(train_X, train_y),
        get_feature_names_from_column_transformer(full_pipeline))

    # Note: we can create a transformer/feature selector that simply drops
    # a specified set of columns. But, we don't do that here to illustrate
    # what to do when transformations don't cleanly fall into the sklearn
    # pattern.

    curated_columns = list(
        set(train_X.columns.to_list()) - set([
            'households', 'longitude', 'total_rooms', 'bedrooms_per_room',
            'population', 'ocean_proximity_INLAND'
        ]))

    train_X = train_X[curated_columns]

    # saving the list of relevant columns
    save_pipeline(
        curated_columns,
        op.abspath(op.join(artifacts_folder, 'curated_columns.joblib')))
    # save the feature pipeline
    save_pipeline(full_pipeline,
                  op.abspath(op.join(artifacts_folder, 'features.joblib')))
