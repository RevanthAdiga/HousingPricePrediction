import pandas as pd
import numpy as np
import pytest


def create_dataframe_values(size):
    float_col1 = pd.Series([1.1*x for x in range(1, size+1)])
    int_col1 = pd.Series([1*x for x in range(1, size+1)])
    string_col1 = pd.Series(['str'+str(x) for x in range(1, 9)])
    string_col2 = pd.Series(np.random.choice(['split1', 'split2', 'split3', np.nan], size).tolist())
    bool_col1 = pd.Series(np.random.choice([True, False], 8).tolist())
    string_col3 = pd.Series(np.random.choice(['Tiger1', 'Tiger2'], size).tolist())
    year_col1 = pd.Series(pd.date_range(start='1990', periods=size).tolist())
    return float_col1, int_col1, string_col1, string_col2, bool_col1, string_col3, year_col1


@pytest.fixture()
def fixture_create_plot_data_ip():
    float_col1, int_col1, string_col1, string_col2, bool_col1, string_col3, year_col1 = create_dataframe_values(8)
    df = pd.DataFrame({'float_col1': float_col1,
                       'int_col1': int_col1,
                       'string_col1': string_col1,
                       'string_col2': string_col2,
                       'bool_col1': bool_col1,
                       'string_col3': string_col3,
                       'year_col1': year_col1})
    return df


@pytest.fixture()
def fixture_create_plot_data_op1():
    """returns the expected results for the 1st yexpr"""
    float_col1, int_col1, string_col1, string_col2, bool_col1, string_col3, year_col1 = create_dataframe_values(8)
    df_columns = ['float_col1', 'bool_col1']
    index_name = 'year_col1'
    size = (8,)
    return df_columns, index_name, size


@pytest.fixture()
def fixture_create_plot_data_op2():
    """returns the expected results for the 2nd yexpr"""
    float_col1, int_col1, string_col1, string_col2, bool_col1, string_col3, year_col1 = create_dataframe_values(8)
    df_columns = ['int_col1', 'string_col1' 'bool_col1']
    index_name = 'year_col1'
    size = (8,)
    return df_columns, index_name, size


@pytest.fixture()
def fixture_create_plot_data_op3(fixture_create_plot_data_ip):
    """returns the expected results for the 3rd yexpr"""
    df = fixture_create_plot_data_ip
    needed_cols = ['int_col1', 'bool_col1','year_col1', 'string_col2']
    groupby_cols = ['bool_col1', 'year_col1', 'string_col2']

    group_results = []
    for function in ['mean', 'sum', 'min', 'max', 'count']:
      grouped_df = df[needed_cols].groupby(groupby_cols).agg(function).reset_index().set_index('year_col1')
      df_columns = grouped_df.columns
      groups = grouped_df['int_col1'].values.tolist()
      group_results.append(groups)
    
    return (df_columns, group_results[0], group_results[1], group_results[2], group_results[3],
            group_results[4])


@pytest.fixture()
def fixture_create_plot_data_op4():
    """returns the expected results for the 4th y expr"""
    df_columns = ['float_col1' 'bool_col1']
    size = (8,)
    return df_columns, size


@pytest.fixture()
def fixture_test_sorter_data():
    """returns the expected results for the sorter"""
    df_columns = ['tigerml_sort_ranking']
    sort_ranking = [[0], [1], [2], [3], [4], [5], [6], [7]]
    return df_columns, sort_ranking
