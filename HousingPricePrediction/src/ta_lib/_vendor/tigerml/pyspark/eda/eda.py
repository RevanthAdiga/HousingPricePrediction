"""Functions to carry out the EDA in spark framework."""

import gc
import logging  # noqa
from collections import Counter

import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hvplot import hvPlot
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import functions as F
from pyspark_dist_explore import distplot

from tigerml.core.plots.bokeh import add_to_secondary, finalize_axes_right
from tigerml.core.reports import create_report
from tigerml.pyspark.core import dp
from tigerml.pyspark.core.utils import (
    append_file_to_path,
    flatten_list,
    time_now_readable,
)

_LOGGER = logging.getLogger(__name__)


def setanalyse(df1, df2, col, simplify=True, exceptions_only=False):
    """
    Given two spark dataframes, returns a dictionary of set analysis.

    A-B: set(A) - set(B)
    B-A: set(B) - set(A)
    AuB: A union B
    A^B: A intersection B

    Parameters
    ----------
        df1, df2: spark dataframes to be evaluated
        exceptions_only: 'False'(default):
            if True, gives only A-B & B-A. False gives all 4.
            True is efficient while dealing with large sets or analyzing exceptions alone.

    """
    A = set(df1.select(col).rdd.map(lambda r: r).collect())
    B = set(df2.select(col).rdd.map(lambda r: r).collect())
    output = {"A-B": A - B, "B-A": B - A}
    if ~exceptions_only:
        output["AuB"] = A.union(B)
        output["A^B"] = A.intersection(B)
    if simplify:
        for key, value in output.items():
            output[key] = len(value)
    return output


# ---------------
# Health Report
# ---------------
def column_values_summary(data):
    """Summarise the column types, number of unique values, and percentage of unique values per column.

    Parameters
    ----------
        data - pyspark.sql.DataFrame

    Returns
    -------
        df - pyspark.sql.DataFrame
    """

    # datatypes
    a = pd.DataFrame({x.name: [x.dataType] for x in list(data.schema)})

    # countDistinct
    b = data.agg(*(F.countDistinct(F.col(c)).alias(c) for c in data.columns)).toPandas()

    # percent of countDistinct over the entire len
    c = round(b * 100 / data.count(), 2)

    df = a.append(b)
    df = df.append(c)

    df.index = ["Data type", "Distinct count", "Distinct count(%)"]
    return df


def get_datatypes(data):
    """List the numerical columns and non_numerical columns.

    Parameters
    ----------
    data: pyspark.sql.DataFrame

    Returns
    -------
    numerical_columns: list(str)
        list of numerical columns
    non_numerical_columns: list(str)
        list of non numerical columns
    """
    numerical_columns = dp.list_numerical_columns(data)
    non_numerical_columns = set(data.columns) - set(numerical_columns)
    return numerical_columns, non_numerical_columns


def get_missing_values_summary(data):
    """Get a summary of the missing values.

    Parameters
    ----------
    data: pyspark.sql.DataFrame

    Returns
    -------
    df_mising: pd.DataFrame
        pyspark dataframe that contains counts of missing values
    """
    df_missing = dp.identify_missing_values(data)
    df_missing = df_missing.toPandas().T
    return df_missing


def _missing_values(data):
    """Get a pandas dataframe with the information about missing values in the dataset.

    Parameters
    ----------
    data: pyspark.sql.DataFrame

    Returns
    -------
    df_missing: pyspark.sql.DataFrame
        pyspark dataframe that contains summary of missing values

    """
    no_of_rows = data.count()
    df = data
    df = get_missing_values_summary(data)
    df = df.reset_index()
    df = df.rename(
        columns=dict(zip(list(df.columns), ["Variable Name", "No of Missing"],))  # noqa
    )
    df["Percentage Missing"] = df["No of Missing"] / float(no_of_rows) * 100
    return df


def get_health_analysis(
    data, missing=True, data_types=True, duplicate_values=True, duplicate_columns=True
):
    """Get the summary of health analysis.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    missing: bool, default is True.
    data_types: bool, default is True.
    duplicate_values:bool, default is True.
    duplicate_columns:bool, default is True.

    Returns
    -------
    dict_summary: dict
        dictionary containing the summary of health analysis
    """
    dict_summary = {
        "data_types": {},
        "missing": {},
        "duplicate_values": {},
        "duplicate_columns": {},
    }
    row_count = data.count()
    numerical_columns, non_numerical_columns = get_datatypes(data)
    numeric_list = [
        len(numerical_columns) / len(data.columns),
        len(non_numerical_columns) / len(data.columns),
    ] * 100
    df_missing = get_missing_values_summary(data)
    per_ = df_missing.sum()[0] / row_count
    missing_values = [1 - per_, per_] * 100
    dict_summary["missing"] = {
        "Available": missing_values[0],
        "Missing": missing_values[1],
    }
    dict_summary["data_types"] = {"Numeric": numeric_list[0], "Others": numeric_list[1]}
    duplicate_rows = (row_count - data.dropDuplicates().count()) / row_count
    duplicate_rows_list = [(1 - duplicate_rows), duplicate_rows]
    dict_summary["duplicate_values"] = {
        "Unique": duplicate_rows_list[0],
        "Duplicated": duplicate_rows_list[1],
    }

    # TBD duplicate columns
    return dict_summary


def plot_health(
    data, missing=True, data_types=True, duplicate_values=True, duplicate_columns=True
):
    """Get the health analysis plots.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    missing: bool, default is True.
    data_types: bool, default is True.
    duplicate_values:bool, default is True.
    duplicate_columns:bool, default is True.

    Returns
    -------
    final_plot: HvPlot
            HvPlot containing all the health plots of variables
    """
    data_dict = get_health_analysis(
        data, missing, data_types, duplicate_values, duplicate_columns
    )

    df_dict = {
        "type": flatten_list([[x] * len(data_dict[x]) for x in data_dict.keys()]),
        "labels": list(data_dict["data_types"].keys())
        + list(data_dict["missing"].keys())
        + list(data_dict["duplicate_values"].keys()),
        "values": list(data_dict["data_types"].values())
        + list(data_dict["missing"].values())
        + list(data_dict["duplicate_values"].values()),
    }
    df = pd.DataFrame(df_dict)
    df = df.set_index(["type", "labels"])
    # this multipliers resolves the issue of duplicate columns as it's values are multiplied by 1 and others
    # with no_of_columns. which was needed for the correct metrics.
    final_plot = None
    for metric in df.index.get_level_values(0).unique():
        plot = (
            hvPlot(df.loc[metric].T)
            .bar(stacked=True, title=metric, height=100, invert=True)
            .opts(xticks=list([i for i in range(df.shape[1])]))
        )
        if final_plot:
            final_plot += plot
        else:
            final_plot = plot
    return final_plot.cols(1)


def missing_plot(data):
    """Get the Missing Variable plot.

    The function returns a bar plot mentioning the number of variables that are present in each missing value
    bucket(0%-100% in increments of 5%).

    Parameters
    ----------
    data: pyspark.sql.DataFrame

    Returns
    -------
    f: `hvplot`
        missing_plot returns a bar plot with the following axis:

        X.axis - % of missing observation bucket
        Y.axis - Number of variables
    """
    # plt.close('all')
    df = data
    missing_values = _missing_values(data)
    break_value = [0, 5, 10, 20, 30, 40, 50, 100]
    lab_value = ["0-5%", "5-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-100%"]
    cuts = pd.cut(
        missing_values["Percentage Missing"],
        bins=break_value,
        labels=lab_value,
        right=True,
    )
    cuts = cuts.value_counts().reindex(lab_value)
    remaining_cols = len(df.columns) - cuts.sum()
    cuts = pd.concat([pd.Series([remaining_cols], index=["No Missing"]), cuts])
    plot = hvPlot(cuts).bar(
        rot=0,
        title="Missing Variables",
        xlabel="# of missing observations",
        ylabel="# of variables",
    )
    return plot


def missing_value_summary(data):
    """Get the summary of missing values computed from `missing_values` function.

    This function describes the share of missing values for each variable
    in the dataset. If there are no missing values, "No Missing Values"
    message is displayed, else a table containing the percentage of
    missing for all variables with missing values are displayed

    Parameters
    ----------
    data: pyspark.sql.DataFrame

    Returns
    -------
    df: pandas.DataFrame

    """
    df = _missing_values(data)
    df = df.loc[df["No of Missing"] != 0].reset_index()
    if df.empty:
        return "No Missing Values"
    else:
        return df


def get_outliers(data, cols=None):
    """To get the summary of outliers.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    cols: list(str)
        list of numerical columns

    Returns
    -------
    outliers_dict_iqr: dict
        contains the details of iqr bounds
    outliers_dict_mean:
        contains the details of std bounds
    """
    iqr_bounds = dp._calculate_outlier_bounds_iqr(data, cols)
    mean_bounds = dp._calculate_outlier_bounds_sdv(data, cols)
    outliers_dict_iqr = {}
    outliers_dict_mean = {}
    df = data
    for col_ in cols:
        df = df.withColumn(
            "lower_bound_iqr",
            F.when(F.col(col_) < iqr_bounds[col_]["min_b"], 1).otherwise(0),
        )
        df = df.withColumn(
            "upper_bound_iqr",
            F.when(F.col(col_) > iqr_bounds[col_]["max_b"], 1).otherwise(0),
        )
        df = df.withColumn(
            "lower_bound_mean",
            F.when(F.col(col_) < mean_bounds[col_]["min_b"], 1).otherwise(0),
        )
        df = df.withColumn(
            "upper_bound_mean",
            F.when(F.col(col_) > mean_bounds[col_]["max_b"], 1).otherwise(0),
        )
        agg_df = (
            df.select(
                "lower_bound_iqr",
                "upper_bound_iqr",
                "lower_bound_mean",
                "upper_bound_mean",
            )
            .groupBy()
            .sum()
            .collect()
        )
        outliers_dict_iqr[col_] = (agg_df[0][0], agg_df[0][1])
        outliers_dict_mean[col_] = (agg_df[0][2], agg_df[0][3])
    return outliers_dict_iqr, outliers_dict_mean


def get_outliers_table(data):
    """To get a dataframe with outlier analysis table.

    Parameters
    ----------
    data: pyspark.sql.DataFrame

    Returns
    -------
    outliers_df: pyspark.sql.DataFrame
    """
    if data is None:
        raise ValueError("Data is not provided")
    outlier_col_labels = [
        "< (mean-3*std)",
        "> (mean+3*std)",
        "< (1stQ - 1.5 * IQR)",
        "> (3rdQ + 1.5 * IQR)",
    ]
    numerical_columns = dp.list_numerical_columns(data)
    outliers_dict_iqr, outliers_dict_mean = get_outliers(data, numerical_columns)
    outliers_df = pd.DataFrame.from_dict(outliers_dict_mean)
    outliers_df = pd.concat([outliers_df, pd.DataFrame.from_dict(outliers_dict_iqr)])
    outliers_df = outliers_df.reset_index(drop=True).T
    outliers_df.rename(
        columns=dict(zip(list(outliers_df.columns), outlier_col_labels)), inplace=True
    )
    outliers_sum = outliers_df.sum(axis=1)
    outliers_df = outliers_df[outliers_sum > 0]
    outliers_df.index.name = "feature"
    return outliers_df


def health_analysis(data, save_as=None, save_path=""):
    """Data health report.

    Compiles outputs from data_health, missing_plot, missing_value_summary and get_outliers_df as a report.

    Parameters
    ----------
    save_as : str, default=None
        give ".html" for saving the report
    save_path : str, default=''
        Location where report to be saved. By default report saved in working directory.

    Examples
    --------
    >>> from tigerml.pyspark import health_analysis
    >>> df = spark.read.parquet("train.parquet")
    >>> health_analysis(df, save_as=".html", save_path="PySpark_Reports/")
    """
    # data_shape = str(data.count())
    health_analysis_report = {}
    health_analysis_report.update({"health_plot": plot_health(data)})
    health_analysis_report.update({"missing_plot": missing_plot(data)})
    health_analysis_report.update(
        {"missing_value_summary": missing_value_summary(data)}
    )

    health_analysis_report.update({"outliers_in_features": get_outliers_table(data)})
    if save_as:
        default_report_name = "health_analysis_report_at_{}".format(time_now_readable())
        save_path = append_file_to_path(save_path, default_report_name + save_as)
        create_report(
            health_analysis_report, path=save_path, format=save_as,
        )
    return health_analysis_report


# ----------------
# Feature Analysis
# ----------------
def describe_data(data, columns):
    """Obtain the basic stats results and percentiles of numerical data.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    columns: list(str)
            the cloumn name list of the numerical variable

    Returns
    -------
    new_df: pyspark.sql.DataFrame
        the numerical describe info. of the input dataframe
    """
    percentiles = [25, 50, 75]
    # array_list=[np.array([row[f"{x}"] for row in data.select(x).collect()],dtype='float') for x in columns]
    temp = data.select(columns).toPandas()
    array_list = [np.array(temp[f"{x}"], dtype="float") for x in columns]
    array_list_samples = [
        list(np.unique(np.unique(row[~np.isnan(row)])[-5:])) for row in array_list
    ]
    array_list_unique = [len(np.unique(row[~np.isnan(row)])) for row in array_list]
    percs = [np.nanpercentile(row, percentiles) for row in array_list]
    percs = np.transpose(percs)
    percs = pd.DataFrame(percs, columns=columns)
    samples = pd.DataFrame([array_list_samples], columns=columns)
    percs = pd.DataFrame([array_list_unique], columns=columns).append(percs)
    percs = samples.append(percs)
    percs["summary"] = ["samples", "nunique"] + [str(p) + "%" for p in percentiles]

    spark_describe = data.describe().toPandas()
    drop_cols = list(set(spark_describe.columns) - set(percs.columns))
    spark_describe.drop(drop_cols, axis=1, inplace=True)
    new_df = pd.concat([spark_describe, percs], ignore_index=True)
    new_df = new_df.round(2)
    new_df = new_df.T
    new_df.columns = list(np.concatenate(new_df.loc[new_df.index == "summary"].values))
    new_df.drop("summary", inplace=True)
    return new_df


def describe_categoricaldata(data, cat_cols):
    """Obtain basic stats results and percentiles of categorical data.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    cat_cols: list(str)
            the cloumn name list of the categorical variable

    Returns
    -------
    new_df: pd.DataFrame
        the categorical describe info. of the input dataframe
    """
    na_list = ["nan", "NA"]
    # array_list=[np.array([row[f"{x}"] for row in data.select(x).collect()],dtype='str') for x in cat_cols]
    temp = data.select(cat_cols).toPandas()
    array_list = [np.array(temp[f"{x}"], dtype="str") for x in cat_cols]
    array_list_samples = [
        list(np.unique(np.unique([val_ for val_ in row if val_ not in na_list])[-5:]))
        for row in array_list
    ]
    array_list_unique = [
        len(np.unique([val_ for val_ in row if val_ not in na_list]))
        for row in array_list
    ]
    samples = pd.DataFrame([array_list_samples], columns=cat_cols)
    unique_df = pd.DataFrame([array_list_unique], columns=cat_cols)
    samples = unique_df.append(samples)
    mode_list = [
        max(dict(Counter(row)), key=dict(Counter(row)).get) for row in array_list
    ]
    samples = samples.append(pd.DataFrame([mode_list], columns=cat_cols))
    mode_freq = [
        dict(Counter(row)).get(mode_) for row, mode_ in zip(array_list, mode_list)
    ]
    samples = samples.append(pd.DataFrame([mode_freq], columns=cat_cols))
    samples["summary"] = ["nunique", "samples", "mode", "mode_freq"]
    samples = samples.T
    samples.columns = list(
        np.concatenate(samples.loc[samples.index == "summary"].values)
    )
    samples.drop("summary", inplace=True)

    return samples


def feature_analysis_table(data):
    """Get the descriptive statistics of the data.

    Parameters
    ----------
    data: pyspark.sql.DataFrame

    Returns
    -------
    numerical_description: pd.DataFrame
        contains the descriptive statistics of numerical variables.
    categorical_description: pd.DataFrame
        contains the descriptive statistics of categorical variables.
    """
    numerical_columns = dp.list_numerical_columns(data)
    numerical_description = describe_data(data, numerical_columns)
    categorical_columns = dp.list_categorical_columns(data)
    categorical_description = describe_categoricaldata(data, categorical_columns)
    return numerical_description, categorical_description


def density_plots_numerical(data):
    """Get the densisty plots of the numerical variables.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    """
    num_cols = dp.list_numerical_columns(data)
    fig, axes = plt.subplots(nrows=int(np.ceil(len(num_cols) / 2)), ncols=2)
    fig.set_size_inches(20, 20)
    axes = np.concatenate(axes)
    plots_dict = {}
    for index_, col_ in enumerate(num_cols):
        plot_ = distplot(axes[index_], [data.select(col_)], bins=40)  # noqa
        axes[index_].set_title(f"distribution of {col_}")
        axes[index_].legend()
        plots_dict.update({col_: plot_})
    return plots_dict


def non_numeric_frequency_plots(data, cols):
    """Get a dictionary of interactive frequency plots and summary table for non-numeric cols.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    cols: list(str)
        default: empty, takes the requested columns in the given dataframe.

    Returns
    -------
    plot: dict
        for all the non-numeric required columns in the list if it is not empty else all non-numeric from the given data.
    """
    plots_dict = {}
    for col_ in sorted(cols):
        series = data.select(col_)
        summary_df = series.describe().toPandas().T.round(2)
        summary_df.columns = list(
            np.concatenate(summary_df.loc[summary_df.index == "summary"].values)
        )
        summary_df.drop("summary", inplace=True)
        summary_table = hvPlot(summary_df).table(
            columns=list(summary_df.columns), height=60, width=600
        )
        freq_plot = hvPlot(
            series.toPandas()[col_].value_counts().head(20).sort_values(ascending=True)
        ).bar(title="Frequency Plot for {}".format(col_), invert=True, width=600)
        plots_dict.update({col_: (freq_plot + summary_table).cols(1)})
    return plots_dict


def density_plots(data, cols):
    """Get a dict of interactive density plots and numeric summary.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    cols: list
        default: empty, takes the requested columns in the given dataframe.

    Returns
    -------
    plots_dict: dict
        for all the requested numeric columns defined in the list if it is not empty else all non-numeric from the given data.
    """
    plots_dict = {}
    for col_ in sorted(cols):
        series = data.select(col_)
        summary_df = series.describe().toPandas().T.round(2)

        summary_df.columns = list(
            np.concatenate(summary_df.loc[summary_df.index == "summary"].values)
        )
        summary_df.drop("summary", inplace=True)
        summary_table = hvPlot(summary_df).table(
            columns=list(summary_df.columns), height=60, width=600
        )
        try:
            hist_plot = hv.Histogram(np.histogram(series.toPandas(), bins=20))
            density_plot = hvPlot(series.toPandas()).kde(
                title="Density Plot for {}".format(col_), width=600
            )
            hooks = [add_to_secondary, finalize_axes_right]
            complete_plot = hist_plot.options(
                color="#00fff0", xlabel=col_
            ) * density_plot.options(hooks=hooks)
            plots_dict[col_] = (complete_plot + summary_table).cols(1)
        except Exception as e:
            plots_dict[col_] = f"Could not generate. Error - {e}"

    return plots_dict


def non_numeric_frequency_plots_v2(data, cols):
    """Get a holoviews layout of interactive frequency plots and summary table for non-numeric cols.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    cols: list(str)
        default: empty, takes the requested columns in the given dataframe.

    Returns
    -------
    plot: holoviews.core.layout.Layout
        for all the non-numeric required columns in the list if it is not empty else all non-numeric from the given data.
    """
    series = data.select(cols)
    summary_df = series.describe().toPandas().T.round(2)
    summary_df.columns = list(
        np.concatenate(summary_df.loc[summary_df.index == "summary"].values)
    )
    summary_df.drop("summary", inplace=True)
    summary_table = hvPlot(summary_df).table(
        columns=list(summary_df.columns), height=60, width=600
    )
    freq_plot = hvPlot(
        series.toPandas()
        .apply(lambda x: x.value_counts().head(20).sort_values(ascending=True))
        .T.reset_index()
        .melt(id_vars="index")
        .dropna()
        .reset_index()
        .drop(columns=["level_0"])
    ).bar(
        invert=True,
        width=600,
        x="variable",
        y=["value"],
        subplots=True,
        by="index",
        shared_axes=False,
    )
    plots = (freq_plot + summary_table).cols(1)
    return plots


def density_plots_v2(data, cols):
    """Get a holoviews layout of interactive density plots.

    A numeric summary for the given columns or all numeric columns for the given data is also provided.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    cols: list
        default: empty, takes the requested columns in the given dataframe.

    Returns
    -------
    plots: holoviews.core.layout.Layout
        for all the requested numeric columns defined in the list if it is not empty else all non-numeric from the given data.
    """
    series = data.select(cols)
    summary_df = series.describe().toPandas().T.round(2)
    summary_df.columns = list(
        np.concatenate(summary_df.loc[summary_df.index == "summary"].values)
    )
    summary_df.drop("summary", inplace=True)
    summary_table = hvPlot(summary_df.loc[cols]).table(
        columns=list(summary_df.columns),
        height=60,
        width=600,
        subplots=True,
        shared_axes=False,
    )

    density_plot = hvPlot(series.toPandas()[cols]).kde(
        width=600, subplots=True, shared_axes=False
    )
    plots = (density_plot + summary_table).cols(1)

    return plots


def feature_density_plots(data, num_cols=[], cat_cols=[]):
    """Get density plots and bar plots for numerical and categorical columns respectively.

    Parameters
    ----------
    data: pyspark.sql.DataFrame

    Returns
    -------
    numerical_plots: dict
        dict containing the density plots of numerical variables.
    categorical_plots:
        dict containing the bar plots of categorical variables

    """
    categorical_cols = dp.list_categorical_columns(data)
    if not cat_cols:
        cat_cols = categorical_cols
    else:
        for col in cat_cols:
            assert (  # noqa
                col in categorical_cols
            ), "{0} is not a valid categorical column in the input data"

    numerical_cols = dp.list_numerical_columns(data)
    if not num_cols:
        num_cols = numerical_cols
    else:
        for col in num_cols:
            assert (  # noqa
                col in numerical_cols
            ), "{0} is not a valid numerical column in the input data"
    numerical_plots = density_plots(data, numerical_cols)
    categorical_plots = non_numeric_frequency_plots(data, categorical_cols)
    return numerical_plots, categorical_plots


def feature_analysis(data, save_as=None, save_path=""):
    """Univariate analysis for the columns.

    Generate summary_stats, distributions and normality tests for columns.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    save_as : str, default=None
        Name of the report. By default name is auto generated from system timestamp.
    save_path : str, default=''
        Location where report to be saved. By default report saved in working directory.

    Examples
    --------
    >>> from tigerml.pyspark.eda import feature_analysis
    >>> df = spark.read.parquet("train.parquet")
    >>> feature_analysis(df, save_as=".html", save_path="PySpark_Reports/")
    """
    report = {}
    numeric_variables, non_numeric_summary = feature_analysis_table(data)
    report["summary_stats"] = {}
    # report['summary_stats']['variable_summary'] = self.variable_summary()
    report["summary_stats"]["numeric_variables"] = [numeric_variables]
    report["summary_stats"]["non_numeric_variables"] = [non_numeric_summary]

    report["distributions"] = {}
    numeric_density, non_numeric_frequency = feature_density_plots(data)
    report["distributions"]["numeric_variables"] = numeric_density
    report["distributions"]["non_numeric_variables"] = non_numeric_frequency

    if save_as:
        default_report_name = "feature_analysis_report_at_{}".format(
            time_now_readable()
        )
        save_path = append_file_to_path(save_path, default_report_name + save_as)
        create_report(
            report, path=save_path, format=save_as,
        )
    return report


# Feature Interactions
def correlation_table(data, plot="table"):
    """Get feature interaction plot or table.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    plot: str
        if table, then correlation table is obtained else correlation plot

    Returns
    -------
    c_df: pd.DataFrame
        correlation table
    heatmap: hvplot
        correlation plot
    """

    cat_cols = dp.list_categorical_columns(data)
    if len(cat_cols):
        data = label_encode(data, cat_cols)
    cols = dp.list_numerical_columns(data)
    assembler = VectorAssembler(
        inputCols=cols, outputCol="features", handleInvalid="skip"
    )
    df_vector = assembler.transform(data).select("features")
    corr_mat = Correlation.corr(df_vector, "features", method="pearson")
    corr_mat = corr_mat.collect()[0].asDict()["pearson(features)"]
    corr_df = pd.DataFrame(corr_mat.toArray())
    corr_df.index, corr_df.columns = cols, cols

    if plot == "table":
        corr_df = corr_df.where(np.triu(np.ones(corr_df.shape)).astype(np.bool))
        c_df = corr_df.stack().reset_index()
        c_df = c_df.rename(
            columns=dict(zip(list(c_df.columns), ["var1", "var2", "corr_coef"]))
        )
        return c_df
    else:
        heatmap = hvPlot(corr_df).heatmap(rot=45, height=450)

        return heatmap


def feature_interactions(data, save_as=None, save_path=""):
    """Feature interactions report.

    Compiles outputs from correlation_table, correlation_heatmap, covariance_heatmap and bivariate_plots as a report.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    save_as : str, default=None
        Name of the report. By default name is auto generated from system timestamp.
    save_path : str, default=''
        Location where report to be saved. By default report saved in working directory.

    Examples
    --------
    >>> from tigerml.pyspark.eda import feature_interactions
    >>> df = spark.read.parquet("train.parquet")
    >>> feature_interactions(df, save_as=".html", save_path="PySpark_Reports/")
    """
    feature_interactions_report = {}
    feature_interactions_report["correlation_table"] = [correlation_table(data)]
    feature_interactions_report["correlation_heatmap"] = [correlation_table(data, "")]

    if save_as:
        default_report_name = "feature_interactions_report_at_{}".format(
            time_now_readable()
        )
        save_path = append_file_to_path(save_path, default_report_name + save_as)
        create_report(
            feature_interactions_report, path=save_path, format=save_as,
        )
    return feature_interactions_report


# ------------
# Key Features
# ------------
def correlation_with_target(data, target_var, cols=None):
    """Get a barplot with correlation with target variable.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    target_var: str
        target variable of the data
    cols: list
        List of numerical columns.
        default - considers all the numeriacal features in the data

    Returns
    -------
    plot: hvplot of bars related to the correlation with target variable
    """
    if not cols:
        cat_cols = dp.list_categorical_columns(data)

        cat_cols = [i for i in cat_cols if i != target_var]
        if len(cat_cols):
            data = label_encode(data, cat_cols)

        cols = dp.list_numerical_columns(data)
    assembler = VectorAssembler(
        inputCols=cols, outputCol="features", handleInvalid="keep"
    )
    df_vector = assembler.transform(data).select("features")
    corr_mat = Correlation.corr(df_vector, "features", method="pearson")
    corr_mat = corr_mat.collect()[0].asDict()["pearson(features)"]
    corr_df = pd.DataFrame(corr_mat.toArray())
    corr_df.index, corr_df.columns = cols, cols
    corr_df = corr_df[[target_var]]
    corr_df = corr_df[corr_df.index != target_var]
    corr_df = corr_df[~corr_df[target_var].isna()]
    corr_df.rename(
        columns={target_var: "Pearson_correlation_with_Target"}, inplace=True
    )
    corr_df.sort_values(by="Pearson_correlation_with_Target", inplace=True)
    plot = hvPlot(corr_df).bar(
        invert=True, title="Feature Correlation with Target Function"
    )
    return plot


def label_encode(data, cat_cols):
    """Obtain label encoding of categorical variables.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    cat_cols: list(str)
        list of categorical column names

    Returns
    -------
    data: pyspark.sql.DataFrame
        dataframe with label encoding of categorical columns
    """
    indexers = []
    for cat_ in cat_cols:
        stringIndexer = StringIndexer(
            inputCol=cat_, outputCol=f"label_encoded_{cat_}", handleInvalid="keep"
        )
        indexers += [stringIndexer]
    pipeline = Pipeline(stages=indexers)
    data = pipeline.fit(data).transform(data)
    return data


def feature_importance(data, target_var, classification=False):
    """Get feature importance based on RandomForestRegressor or RandomForestClassifier.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    target_var: str
        target variable of the data
    classification: bool, default is False.
        choosen basen on Regression or Classification problem.

    Returns
    -------
    plot: hvplot
        feature importance plot based on Random Forests
    """
    cat_cols = dp.list_categorical_columns(data)

    if len(cat_cols):
        data = label_encode(data, cat_cols)
    num_cols = dp.list_numerical_columns(data)
    if classification:
        from pyspark.ml.classification import RandomForestClassifier

        if target_var in cat_cols:
            data = data.withColumnRenamed(f"label_encoded_{target_var}", "target")
        else:
            print(True)
            data = data.withColumnRenamed(f"{target_var}", "target")
        rf = RandomForestClassifier(
            numTrees=3, maxDepth=20, labelCol="target", maxBins=100, seed=42
        )
    else:
        from pyspark.ml.regression import RandomForestRegressor

        data = data.withColumnRenamed(f"{target_var}", "target")
        # Load model
        rf = RandomForestRegressor(
            numTrees=3, maxDepth=20, labelCol="target", maxBins=100, seed=42
        )

    if target_var in num_cols:
        num_cols.remove(target_var)
    elif f"label_encoded_{target_var}" in num_cols:
        num_cols.remove(f"label_encoded_{target_var}")
    assembler = VectorAssembler(
        inputCols=num_cols, outputCol="features", handleInvalid="skip"
    )
    model_data = assembler.transform(data)
    model_data = model_data.select(["features", "target"])

    # BUILD THE MODEL
    model = rf.fit(model_data)

    # FEATURE IMPORTANCES
    feature_importance = pd.DataFrame.from_dict(
        dict(zip(num_cols, model.featureImportances.toArray())), orient="index"
    ).rename(columns={0: "Feature Importance"})
    feature_importance.sort_values(by="Feature Importance", inplace=True)
    plot = hvPlot(feature_importance).bar(
        invert=True, title="Feature Importances from RF"
    )
    return plot


def feature_analysis_pca(data, target_var):
    """Get feature importance based on RandomForestRegressor or RandomForestClassifier.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    target_var: str
        target variable of the data

    Returns
    -------
    transformed: pyspark.sql.DataFrame
        modified dataframe with pca column
    """
    from pyspark.ml.feature import PCA as PCAml

    cat_cols = dp.list_categorical_columns(data)
    cat_cols = [i for i in cat_cols if i != target_var]
    if len(cat_cols):
        data = label_encode(data, cat_cols)
    num_cols = dp.list_numerical_columns(data)
    num_cols = [i for i in num_cols if i != target_var]
    assembler = VectorAssembler(
        inputCols=num_cols, outputCol="features", handleInvalid="skip"
    )
    model_data = assembler.transform(data)
    pca = PCAml(k=2, inputCol="features", outputCol="pca")
    model = pca.fit(model_data)
    transformed = model.transform(model_data)
    return transformed


class EDAReportPyspark:
    """EDA toolkit for classification and regression models.

    To evaluate and generate reports to summarize, data health, univariate & bivariate analyis, interactions and keydrivers.

    Parameters
    ----------
    data: pyspark.sql.DataFrame

    y : string, default=None
        Name of the target column

    is_classification : bool, default=None
        Set to True, for classificaiton target

    Examples
    --------
    >>> from tigerml.pyspark.eda import EDAReportPyspark
    >>> anp = EDAReportPyspark(data=train_df, is_classification=True, y="target_var")
    >>> anp.get_report(y="target_var",save_path="PySpark_Reports/")
    """

    def __init__(self, data, is_classification, y=None):
        self.data = data
        self.y = y
        self.is_classification = is_classification

    def _set_y_cols(self, y=None):
        if isinstance(y, str):
            y = [y]
        if y:
            return y
        else:
            return []

    def _get_x_cols(self, y=None):
        data = self.data
        if y is not None:
            y = [y] if isinstance(y, str) else y
        else:
            y = self.y_cols
        if self.y_cols:
            return [col for col in data.columns if col not in y]
        else:
            return list(data.columns)

    def _set_xy_cols(self, y):
        self.y_cols = self._set_y_cols(y=y)
        self.x_cols = self._get_x_cols()

    def key_drivers(
        self, y=None, features=None, quick=True, save_as=None, save_path=""
    ):
        """Univariate analysis for the columns.

        Generate summary_stats, distributions and normality tests for columns.

        Parameters
        ----------
        y : Target column name (String)
        features : list, default=[]
            list of columns in the dataframe for analysis. By default all are used.
        save_as : str, default=None
            Name of the report. By default name is auto generated from system timestamp.
        save_path : str, default=''
            Location where report to be saved. By default report saved in working directory.

        Examples
        --------
        >>> from tigerml.pyspark.eda import feature_interactions
        >>> anp = EDAReportPyspark(data=train_df, is_classification=True, y="target_var")
        >>> anp.key_drivers(y="target_var", save_as=".html", save_path="PySpark_Reports/")
        """
        if y:
            assert isinstance(y, str) or isinstance(y, list)
            ys = y if isinstance(y, list) else [y]
            self._set_xy_cols(ys)
        else:
            raise Exception("dependent variable name needs to be passed")

        if features:
            features = list(set(features) & set(self.x_cols))
        else:
            features = self.x_cols

        key_drivers = {}
        for y in ys:
            key_drivers[y] = {}
            key_drivers[y]["feature_scores"] = correlation_with_target(self.data, y)
            key_drivers[y]["feature_importances"] = feature_importance(
                self.data, y, classification=self.is_classification
            )
            # key_drivers[y]["pca_analysis"] = self.get_pca_analysis(features=features)

        # key_drivers[y]['tsne_projection'] = self.get_tsne_projection()
        if save_as:
            default_report_name = "key_drivers_report_at_{}".format(time_now_readable())
            save_path = append_file_to_path(save_path, default_report_name + save_as)
            create_report(
                key_drivers, path=save_path, format=save_as, split_sheets=True,
            )
        self.key_drivers_report = key_drivers
        return key_drivers

    def _create_report(self, y=None, quick=True, corr_threshold=None):
        if y:
            self._set_xy_cols(y)
        self.report = {}
        self.report["data_preview"] = {"head": [self.data.limit(5).toPandas()]}
        self.report["health_analysis"] = health_analysis(self.data)
        # self.report['data_preview']['pre_processing'] = self._prepare_data(corr_threshold)
        self.report["feature_analysis"] = feature_analysis(self.data)
        self.report["feature_interactions"] = feature_interactions(self.data)
        if self.y_cols:
            self.report["key_drivers"] = self.key_drivers(quick=quick, y=self.y_cols)
        else:
            _LOGGER.info(
                "Could not generate key drivers report as dependent variable is not defined"
            )

    def _save_report(self, format=".html", name="", save_path="", tiger_template=False):
        if not name:
            name = "data_exploration_report_at_{}".format(time_now_readable())
        create_report(
            self.report,
            name=name,
            path=save_path,
            format=format,
            split_sheets=True,
            tiger_template=tiger_template,
        )
        del self.report
        gc.collect()

    def get_report(
        self,
        format=".html",
        name="",
        y=None,
        corr_threshold=None,
        quick=True,
        save_path="",
        tiger_template=False,
    ):
        """Create consolidated report on data preview,feature analysis,feature interaction and health analysis.

        The consolidated report also includes key driver report if y(target dataframe) is passed while
        calling create_report.

        Parameters
        ----------
        y : str, default = None
        format : str, default='.html'
            format of report to be generated. possible values '.xlsx', '.html'
        name : str, default=None
            Name of the report. By default name is auto generated from system timestamp.
        save_path : str, default=''
            location with filename where report to be saved. By default is auto generated from system timestamp and saved in working directory.
        quick : boolean, default=True
            If true,calculate SHAP values and create bivariate plots
        corr_threshold : float, default=None
            To specify correlation threshold
        """
        self._create_report(y=y, quick=quick, corr_threshold=corr_threshold)
        return self._save_report(
            format=format, name=name, save_path=save_path, tiger_template=tiger_template
        )
