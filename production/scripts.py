"""Module for listing down additional custom functions required for production."""

import numpy as np
import pandas as pd


def binned_income_cat(df):
    """Bin the median income column using quantiles."""
    return pd.cut(df["median_income"],
                  bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                  labels=[1, 2, 3, 4, 5])
