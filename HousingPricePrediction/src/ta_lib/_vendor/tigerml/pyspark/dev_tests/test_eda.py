import pandas as pd
from hypothesis import given, settings
from hypothesis.extra.pandas import columns, data_frames, range_indexes, column
from hypothesis.strategies import dates, floats, integers, booleans, tuples, sampled_from, composite, data, lists
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, DataType, IntegerType, StringType, TimestampType, LongType, FloatType, StructField, Row
from tigerml.pyspark.eda.eda import setanalyse, get_missing_values_summary, column_values_summary, get_datatypes, _missing_values, get_outliers_table, describe_categoricaldata
from tigerml.pyspark.core.dp import list_numerical_columns, list_categorical_columns
# import findspark
# findspark.init()  # to make sure SPARK_HOME env variable is set correctly

# import os
# os.environ['SPARK_HOME'] = "D:/GitHub_AR/pyspark_trials/Installations/spark-3.0.1"  # path to Spark binaries
# os.environ['HADOOP_HOME'] = "D:/GitHub_AR/pyspark_trials/Installations/hadoop27"


@composite
def df_with_required_cols(draw, required_cols='all', non_empty_row='have', have_na=False):
    """ An hypothesis.composite strategy for creating generic pandas_df."""
    if required_cols == 'all':
        required_cols = ['int_col', 'float_col', 'cat_col', 'dt_col', 'bool_col']
    date_options = [pd.Timestamp('1/1/1995'), pd.Timestamp('1/1/2005'), pd.Timestamp('1/1/2015')]
    float_options = floats(allow_infinity=False, allow_nan=False)
    string_options = ['red', 'Blue', 'GREEN']
    bool_options = [True, False]
    if have_na:
        date_options += [pd.NaT]
        float_options = floats(allow_infinity=False, allow_nan=True)
        string_options += ['NA', None]
        bool_options += [None]

    column_list = []
    if 'int_col' in required_cols:
        column_list += [column('int_col', elements=integers())]
    if 'int_col2' in required_cols:
        column_list += [column('int_col2', elements=integers())]
    if 'float_col' in required_cols:
        # column_list += [column('float_col', elements=floats(allow_infinity=True, allow_nan=True))]
        column_list += [column('float_col', elements=float_options)]
    if 'float_col2' in required_cols:
        column_list += [column('float_col2', elements=float_options)]
    if 'cat_col' in required_cols:
        column_list += [column('cat_col', elements=sampled_from(string_options))]
    if 'cat_col2' in required_cols:
        column_list += [column('cat_col2', elements=sampled_from(string_options))]
    if 'dt_col' in required_cols:
        column_list += [column('dt_col', elements=sampled_from(date_options))]
    if 'dt_col2' in required_cols:
        column_list += [column('dt_col2', elements=sampled_from(date_options))]
    if 'bool_col' in required_cols:
        column_list += [column('bool_col', elements=sampled_from(bool_options))]
    if 'bool_col2' in required_cols:
        column_list += [column('bool_col2', elements=sampled_from(bool_options))]

    df = draw(data_frames(index=range_indexes(min_size=3), columns=column_list))
    if non_empty_row == 'have':
        non_empty_row = True
    else:
        non_empty_row = False  # draw(sampled_from([True, False]))
    if non_empty_row:
        additional_values = {'int_col': [57, 24, 32], 'int_col2': [67, 34, 42],
                             'float_col': [5.7, 2.4, 3.2], 'float_col2': [6.7, 3.4, 4.2],
                             'cat_col': ['red', 'Blue', 'GREEN'], 'cat_col2': ['red', 'Blue', 'GREEN'],
                             'dt_col': [pd.Timestamp('1/1/1965'), pd.Timestamp('1/1/1975'), pd.Timestamp('1/1/1985')],
                             'dt_col2': [pd.Timestamp('1/1/1965'), pd.Timestamp('1/1/1975'), pd.Timestamp('1/1/1985')],
                             'bool_col': [True, False, False], 'bool_col2': [True, False, False]}

        for i in range(3):
            new_row = {}
            for col in required_cols:
                new_row[col] = additional_values[col][i]
            df = df.append(pd.Series(new_row), ignore_index=True)
    return df


def pandas_to_spark(spark, pandas_df):
    """ Will return a spark dataframe for given pandas dataframe."""
    def equivalent_type(f):
        if f == 'datetime64[ns]':
            return TimestampType()
        elif f == 'int64':
            return LongType()
        elif f == 'int32':
            return IntegerType()
        elif f == 'float64':
            return FloatType()
        else:
            return StringType()

    cols = list(pandas_df.columns)
    types = list(pandas_df.dtypes)
    struct_list = []
    for col, type_ in zip(cols, types):
        struct_list.append(StructField(col, equivalent_type(type_)))
    p_schema = StructType(struct_list)
    return spark.createDataFrame(pandas_df, p_schema)


# Function to give meaningful test_ids for all the test combinations
def id_func(param):
    id_dict = {'T1': 'DfShape_test', 'T2': 'ListLength_test', 'T3': 'DfValue_test', 'True': 'Columns_with_na',
               'False': 'Columns_without_na', 'Only_nums_df': 'Only_nums_df', 'Only_cat_df': 'Only_cat_df'}
    return id_dict[str(param)]


# Fixture for assigning the type of feature columns (called by @pytest.mark.parametrize in each test case)
@pytest.fixture(params=['Mixed_df', 'Only_nums_df', 'Only_cat_df', 'Only_dt_df', 'Only_bool_df'])
def df_type(request):
    df_type_dict = {'Mixed_df': ['int_col', 'float_col', 'cat_col', 'dt_col', 'bool_col'],
                    'Mixed_df2': ['int_col', 'float_col', 'cat_col', 'dt_col', 'bool_col',
                                  'int_col2', 'float_col2', 'cat_col2', 'dt_col2', 'bool_col2'],
                    'Only_nums_df': ['int_col', 'float_col'], 'Only_cat_df': ['cat_col'],
                    'Only_dt_df': ['dt_col'], 'Only_bool_df': ['bool_col']}
    return df_type_dict[request.param]


@pytest.mark.parametrize("n_partitions",
                         ["n_partitions1", "n_partitions2"])
@pytest.mark.parametrize("n_threads",
                         ["n_threads2", "n_threads3", "n_threads4"])
@pytest.mark.parametrize("test_type",
                         ["T3"],
                         ids=id_func)
@settings(max_examples=10, deadline=None)
@given(test_df=df_with_required_cols(required_cols=['int_col', 'float_col']))
def test_setanalyse(test_df, n_threads, n_partitions, test_type):
    """ test for setanalyse function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = SparkSession.builder.master(threads_str).appName("pyspark.eda_local_testing").getOrCreate()
    test_sdf = pandas_to_spark(spark_session, test_df)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    results = setanalyse(test_sdf, test_sdf, test_sdf.columns)
    if test_type == "T1":
        assert results['A-B'] + results['B-A'] == 0


@pytest.mark.parametrize("n_partitions",
                         ["n_partitions1", "n_partitions2"])
@pytest.mark.parametrize("n_threads",
                         ["n_threads2", "n_threads3", "n_threads4"])
@pytest.mark.parametrize("test_type",
                         ["T1"],
                         ids=id_func)
@settings(max_examples=10, deadline=None)
@given(test_df=df_with_required_cols(required_cols=['int_col', 'float_col']))
def test_column_values_summary(test_df, n_threads, n_partitions, test_type):
    """ test for column_values_summary function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = SparkSession.builder.master(threads_str).appName("pyspark.eda_local_testing").getOrCreate()
    test_sdf = pandas_to_spark(spark_session, test_df)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    results = column_values_summary(test_sdf)
    if test_type == "T1":
        assert results.shape == (3, test_df.shape[1])


@pytest.mark.parametrize("n_partitions",
                         ["n_partitions1", "n_partitions2"])
@pytest.mark.parametrize("n_threads",
                         ["n_threads2", "n_threads3", "n_threads4"])
@pytest.mark.parametrize("test_type",
                         ["T2"],
                         ids=id_func)
@settings(max_examples=10, deadline=None)
@given(test_df=df_with_required_cols(required_cols=['int_col', 'float_col']))
def test_get_datatypes(test_df, n_threads, n_partitions, test_type):
    """ test for get_datatypes function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = SparkSession.builder.master(threads_str).appName("pyspark.eda_local_testing").getOrCreate()
    test_sdf = pandas_to_spark(spark_session, test_df)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    results = get_datatypes(test_sdf)
    if test_type == "T2":
        assert len(results[0]) + len(results[1]) == test_df.shape[1]


@pytest.mark.parametrize("n_partitions",
                         ["n_partitions1", "n_partitions2"])
@pytest.mark.parametrize("n_threads",
                         ["n_threads2", "n_threads3", "n_threads4"])
@pytest.mark.parametrize("test_type",
                         ["T2"],
                         ids=id_func)
@pytest.mark.parametrize("cols_have_na",
                         [False, True],
                         ids=id_func)
@pytest.mark.parametrize("df_type",
                         ['Only_nums_df'],
                         ids=id_func, indirect=True)
@settings(max_examples=10, deadline=None)
@given(test_df=data())
def test_get_missing_values_summary(test_df, n_threads, n_partitions, test_type, cols_have_na, df_type):
    """ test for missing_values_summary function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = SparkSession.builder.master(threads_str).appName("pyspark.eda_local_testing").getOrCreate()
    test_df_ = test_df.draw(df_with_required_cols(required_cols=df_type, have_na=cols_have_na))
    test_sdf = pandas_to_spark(spark_session, test_df_)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    results = get_missing_values_summary(test_sdf)
    if cols_have_na:
        assert results[0].sum() >= 0
    else:
        assert results[0].sum() == 0


@pytest.mark.parametrize("n_partitions",
                         ["n_partitions1", "n_partitions2"])
@pytest.mark.parametrize("n_threads",
                         ["n_threads2", "n_threads3", "n_threads4"])
@pytest.mark.parametrize("test_type",
                         ["T2"],
                         ids=id_func)
@pytest.mark.parametrize("cols_have_na",
                         [False, True],
                         ids=id_func)
@pytest.mark.parametrize("df_type",
                         ['Only_nums_df'],
                         ids=id_func, indirect=True)
@settings(max_examples=10, deadline=None)
@given(test_df=data())
def test_missing_values(test_df, n_threads, n_partitions, test_type, cols_have_na, df_type):
    """ test for _missing_values function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = SparkSession.builder.master(threads_str).appName("pyspark.eda_local_testing").getOrCreate()
    test_df_ = test_df.draw(df_with_required_cols(required_cols=df_type, have_na=cols_have_na))
    test_sdf = pandas_to_spark(spark_session, test_df_)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    results = _missing_values(test_sdf)
    results_list = results['Percentage Missing'].values.tolist()
    if cols_have_na:
        for val in results_list:
            assert 0 <= val <= 100
    else:
        for val in results_list:
            assert val == 0


@pytest.mark.parametrize("n_partitions",
                         ["n_partitions1", "n_partitions2"])
@pytest.mark.parametrize("n_threads",
                         ["n_threads2", "n_threads3", "n_threads4"])
@pytest.mark.parametrize("test_type",
                         ["T2"],
                         ids=id_func)
@settings(max_examples=10, deadline=None)
@given(test_df=df_with_required_cols(required_cols=['int_col', 'float_col']))
def test_get_outliers_table(test_df, n_threads, n_partitions, test_type):
    """ test for get_outliers_table function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = SparkSession.builder.master(threads_str).appName("pyspark.eda_local_testing").getOrCreate()
    test_sdf = pandas_to_spark(spark_session, test_df)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    results = get_outliers_table(test_sdf)
    if test_type == "T2":
        assert results.columns.tolist() == ['< (mean-3*std)', '> (mean+3*std)',
                                            '< (1stQ - 1.5 * IQR)', '> (3rdQ + 1.5 * IQR)']


@pytest.mark.parametrize("n_partitions",
                         ["n_partitions1", "n_partitions2"])
@pytest.mark.parametrize("n_threads",
                         ["n_threads2", "n_threads3", "n_threads4"])
@pytest.mark.parametrize("test_type",
                         ["T2"],
                         ids=id_func)
@pytest.mark.parametrize("df_type",
                         ['Only_cat_df'],
                         ids=id_func, indirect=True)
@settings(max_examples=10, deadline=None)
@given(test_df=data())
def test_describe_categoricaldata(test_df, n_threads, n_partitions, test_type, df_type):
    """ test for describe_categoricaldata function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = SparkSession.builder.master(threads_str).appName("pyspark.eda_local_testing").getOrCreate()
    test_df_ = test_df.draw(df_with_required_cols(required_cols=df_type))
    test_sdf = pandas_to_spark(spark_session, test_df_)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    results = describe_categoricaldata(test_sdf, cat_cols=list_categorical_columns(test_sdf))
    if test_type == "T2":
        assert results.columns.tolist() == ['nunique', 'samples', 'mode', 'mode_freq']
