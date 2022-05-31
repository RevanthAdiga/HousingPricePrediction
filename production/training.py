"""Processors for the model training step of the worklow."""
import logging
import os.path as op
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from ta_lib.core.api import (
    DEFAULT_ARTIFACTS_PATH,
    get_dataframe,
    get_feature_names_from_column_transformer,
    load_dataset,
    load_pipeline,
    register_processor,
    save_pipeline,
)

logger = logging.getLogger(__name__)


@register_processor("model-gen", "train-model")
def train_model(context, params):
    """Train a regression model."""
    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    input_features_ds = "train/housing/features"
    input_target_ds = "train/housing/target"

    # load training datasets
    train_X = load_dataset(context, input_features_ds)
    train_y = load_dataset(context, input_target_ds)

    # load pre-trained feature pipelines and other artifacts
    curated_columns = load_pipeline(
        op.join(artifacts_folder, "curated_columns.joblib"))
    full_pipeline = load_pipeline(op.join(artifacts_folder, "features.joblib"))

    # sample data if needed. Useful for debugging/profiling purposes.
    sample_X = train_X.sample(frac=0.1, random_state=context.random_seed)
    sample_y = train_y.loc[sample_X.index]
    sample_train_X = get_dataframe(
        full_pipeline.fit_transform(sample_X, sample_y),
        get_feature_names_from_column_transformer(full_pipeline),
    )
    # nothing to do for target
    sample_train_y = sample_y

    print(sample_train_X, sample_train_y)

    # transform the training data
    train_X = get_dataframe(
        full_pipeline.fit_transform(train_X, train_y),
        get_feature_names_from_column_transformer(full_pipeline),
    )
    train_X = train_X[curated_columns]

    # create training pipeline
    forest_reg = Pipeline([('forest_reg',
                            RandomForestRegressor(max_features=4,
                                                  n_estimators=30,
                                                  random_state=42))])

    # fit the training pipeline
    forest_reg.fit(train_X, train_y.values.ravel())

    # save fitted training pipeline
    save_pipeline(
        forest_reg,
        op.abspath(op.join(artifacts_folder, "train_pipeline.joblib")))
