import numpy as np
import pandas as pd
import pytest
import tigerml.core.dataframe as td
from datetime import timedelta
from tigerml.viz.data_exploration import DataExplorer
from tigerml.viz.widget.components.filters.constants import *
from hypothesis import given, settings
from hypothesis.extra.pandas import columns, data_frames, range_indexes, column
from hypothesis.strategies import dates, floats, integers, booleans, tuples, sampled_from, composite, data, lists


@composite
def df_with_required_cols(draw, required_cols='all', non_empty_row='have', have_na=False):
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


# Function to give meaningful test_ids for all the test combinations
def id_func(param):
    id_dict = {'1': '1_lev_segment', '2': '2_lev_segment', '3': '3_lev_segment',
               'True': 'Columns_with_na', 'False': 'Columns_without_na',
               'Mixed_df': 'Mixed_df', 'Only_nums_df': 'Only_nums_df', 'Mixed_df2': 'Mixed_df2',
               'Only_cat_df': 'Only_cat_df', 'Only_dt_df': 'Only_dt_df',
               'all': 'all',
               'T1': 'Test_for_return', 'T2': 'Test_for_UniqueSegments', 'T3': 'Test_for_NanValues',
               'I1': 'child_parent_assign', 'I2': 'check_has_changes', 'I3': 'children_types',
               'I4': 'dtype_assignment', 'I5': 'filter_options',
               'Q_T': 'quick=True', 'Q_F': 'quick=False',
               'yes': 'Shuffled_segments', 'no': 'Sorted_segments',
               'have': 'Columns_have_NonEmptyRow', 'not_have': 'Columns_not_have_NonEmptyRow',
               'C1': 'One_filter_condition', 'C2': 'Two_filter_condition'}
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


@pytest.mark.parametrize("cols_have_na",
                         [False, True],
                         ids=id_func)
@pytest.mark.parametrize("have_non_empty_row",
                         ['have', 'not_have'],
                         ids=id_func)
@pytest.mark.parametrize("df_type",
                         ['Mixed_df'],
                         ids=id_func, indirect=True)
@pytest.mark.parametrize("check_type",
                         ['I1', 'I2', 'I3'],  # 'I4' has to be debugged
                         ids=id_func)
@settings(max_examples=10, deadline=None)
@given(test_df=data())
def test_FilterPanel_state(test_df, check_type, df_type, have_non_empty_row, cols_have_na):
    test_tdf = td.DataFrame(test_df.draw(df_with_required_cols(required_cols=df_type, non_empty_row=have_non_empty_row,
                                                               have_na=cols_have_na)))
    test_tdf = test_tdf.categorize()
    de_obj = DataExplorer(test_tdf)
    if check_type == 'I1':
        # Test if children and parent are assigned rightly
        assert de_obj.filters == de_obj.children[2] and de_obj.filters.parent == de_obj
    elif check_type == 'I2':
        assert de_obj.filters.has_changes is False
    elif check_type == 'I3':
        assert 'FilterPanel' in str(type(de_obj.filters)) and 'None' in str(type(de_obj.filters.filter_group))
        de_obj.filters.show()
        de_obj.filters._initiate_filters()
        assert 'FilterWrapper' in str(type(de_obj.filters.filter_group))
    else:
        # Check for dtypes
        expected_dtypes = {'int_col': 'numeric', 'int_col2': 'numeric', 'float_col': 'numeric', 'float_col2': 'numeric',
                           'dt_col': 'datetime', 'dt_col2': 'datetime', 'bool_col': 'bool', 'bool_col2': 'bool',
                           'cat_col': 'category', 'cat_col2': 'category'}
        for col in de_obj.data.columns:
            assert de_obj.filters.dypes[col] == expected_dtypes[col]


@pytest.mark.parametrize("cols_have_na",
                         [False, True],
                         ids=id_func)
@pytest.mark.parametrize("have_non_empty_row",
                         ['have', 'not_have'],
                         ids=id_func)
@pytest.mark.parametrize("df_type",
                         ['Mixed_df'],
                         ids=id_func, indirect=True)
@pytest.mark.parametrize("check_type",
                         ['I1', 'I2', 'I3'],  # 'I5' has to be debugged
                         ids=id_func)
@settings(max_examples=10, deadline=None)
@given(test_df=data())
def test_filter_options(test_df, check_type, df_type, have_non_empty_row, cols_have_na):
    test_tdf = td.DataFrame(test_df.draw(df_with_required_cols(required_cols=df_type, non_empty_row=have_non_empty_row,
                                                               have_na=cols_have_na)))
    test_tdf = test_tdf.categorize()
    de_obj = DataExplorer(test_tdf)
    de_obj.filters.show()
    de_obj.filters._initiate_filters()
    # de_obj.filters.filter_group.add_filter()
    if check_type == 'I1':
        # Test if children and parent are assigned rightly
        assert len(de_obj.filters.filter_group.children) == 1 and \
               de_obj.filters.filter_group.children[0].parent == de_obj.filters.filter_group
    elif check_type == 'I2':
        assert de_obj.filters.has_changes is True
    elif check_type == 'I3':
        assert 'FilterCondition' in str(type(de_obj.filters.filter_group.children[0]))
    else:
        # Check for dtypes
        expected_options_dict = {'int_col': DTYPES.numeric, 'int_col2': DTYPES.numeric,
                                 'float_col': DTYPES.numeric, 'float_col2': DTYPES.numeric,
                                 'cat_col': DTYPES.category, 'cat_col2': DTYPES.category,
                                 'dt_col': DTYPES.datetime, 'dt_col2': DTYPES.datetime,
                                 'bool_col': DTYPES.bool, 'bool_col2': DTYPES.bool}
        for col in df_type:
            de_obj.filters.filter_group.children[0].select_col.value = col
            assert de_obj.filters.filter_group.children[0].condition.options == CONDITIONS[expected_options_dict[col]]


@pytest.mark.parametrize("cols_have_na",
                         [False, True],
                         ids=id_func)
@pytest.mark.parametrize("have_non_empty_row",
                         ['have', 'not_have'],
                         ids=id_func)
@pytest.mark.parametrize("df_type",
                         ['Mixed_df'],
                         ids=id_func, indirect=True)
@pytest.mark.parametrize("check_type",
                         ['I2', 'I3'],  # 'I1', 'I2', 'I3', 'I5' has to be debugged
                         ids=id_func)
@settings(max_examples=10, deadline=None)
@given(test_df=data())
def test_filtergroup_options(test_df, check_type, df_type, have_non_empty_row, cols_have_na):
    test_tdf = td.DataFrame(test_df.draw(df_with_required_cols(required_cols=df_type, non_empty_row=have_non_empty_row,
                                                               have_na=cols_have_na)))
    test_tdf = test_tdf.categorize()
    de_obj = DataExplorer(test_tdf)
    de_obj.filters.show()
    de_obj.filters._initiate_filters()
    de_obj.filters.filter_group.convert_to_group()
    if check_type == 'I1':
        # Test if children and parent are assigned rightly
        assert len(de_obj.filters.filter_group.children) == 2 and \
               de_obj.filters.filter_group.children[0].parent == de_obj.filters.filter_group and \
               len(de_obj.filters.filter_group.children[0].children) == 1 and \
               de_obj.filters.filter_group.children[0].children[0].parent == de_obj.filters.filter_group.children[0]
    elif check_type == 'I2':
        assert de_obj.filters.has_changes is True
    elif check_type == 'I3':
        assert 'FilterWrapper' in str(type(de_obj.filters.filter_group.children[0])) and \
               'FilterCondition' in str(type(de_obj.filters.filter_group.children[0].children[0]))
    else:
        # Check for dtypes
        expected_options_dict = {'int_col': DTYPES.numeric, 'int_col2': DTYPES.numeric,
                                 'float_col': DTYPES.numeric, 'float_col2': DTYPES.numeric,
                                 'cat_col': DTYPES.category, 'cat_col2': DTYPES.category,
                                 'dt_col': DTYPES.datetime, 'dt_col2': DTYPES.datetime,
                                 'bool_col': DTYPES.bool, 'bool_col2': DTYPES.bool}
        for col in df_type:
            de_obj.filters.filter_group.children[0].children[0].select_col.value = col
            assert de_obj.filters.filter_group.children[0].children[0].condition.options == \
                   CONDITIONS[expected_options_dict[col]]


@pytest.mark.parametrize("cols_have_na",
                         [False, True],
                         ids=id_func)
@pytest.mark.parametrize("have_non_empty_row",
                         ['have'],
                         ids=id_func)
@pytest.mark.parametrize("df_type",
                         ['Mixed_df'],
                         ids=id_func, indirect=True)
@pytest.mark.parametrize("check_type",
                         ['C1'],  # 'C2' should be debugged
                         ids=id_func)
@pytest.mark.parametrize("select_col",
                         ['int_col', 'float_col', 'cat_col', 'bool_col'])  # 'dt_col' should be debugged
@settings(max_examples=10, deadline=None)
@given(test_df=data(), condition=data(), input_val=data(), combiner=data())
def test_filter_condition(test_df, select_col, condition, input_val, combiner, df_type, have_non_empty_row,
                          cols_have_na, check_type):
    test_tdf = td.DataFrame(test_df.draw(df_with_required_cols(required_cols=df_type, non_empty_row=have_non_empty_row,
                                                               have_na=cols_have_na)))
    test_tdf = test_tdf.convert_datetimes()
    test_tdf = test_tdf.categorize()
    test_tdf = test_tdf.order_categories()
    if check_type == 'C1':
        de_obj = DataExplorer(test_tdf)
        de_obj.filters.show()
        de_obj.filters._initiate_filters()
    elif check_type == 'C2':
        de_obj = DataExplorer(test_tdf)
        de_obj.filters.show()
        de_obj.filters._initiate_filters()
        de_obj.filters.filter_group.add_filter()
        de_obj.filters.filter_group.combiner.value = combiner.draw(
            sampled_from(de_obj.filters.filter_group.combiner.values))
    for child in de_obj.filters.filter_group.children:
        child.select_col.value = select_col  # .draw(sampled_from(child.select_col.values))
        child.condition.value = condition.draw(sampled_from(child.condition.values))
        input_value = input_val.draw(sampled_from(sorted(test_tdf[child.select_col.value].dropna().values)))
        if isinstance(child.input_val.value, list):  # for categorical, boolean or <5% unique values columns
            input_value = [input_value]
        elif isinstance(child.input_val.value, tuple):  # for datetime columns (a tuple of 2 values would require)
            input_value2 = input_val.draw(sampled_from(
                test_tdf[test_tdf[child.select_col.value] >= input_value][child.select_col.value].dropna().values))
            input_value = (input_value, input_value2)
        else:  # for numerical columns
            input_value = str(input_value)
        child.input_val.value = input_value
    filter_series = de_obj.filters.get_filters()
    assert isinstance(filter_series.sum(), np.number)


@pytest.mark.parametrize("cols_have_na",
                         [False, True],
                         ids=id_func)
@pytest.mark.parametrize("have_non_empty_row",
                         ['have'],
                         ids=id_func)
@pytest.mark.parametrize("df_type",
                         ['Mixed_df2'],
                         ids=id_func, indirect=True)
@settings(max_examples=10, deadline=None)
@given(test_df=data(), select_col=data(), condition=data(), input_val=data(), combiner=data())
def test_filter_group_condition(test_df, select_col, condition, input_val, combiner, df_type, have_non_empty_row,
                                cols_have_na):
    test_tdf = td.DataFrame(test_df.draw(df_with_required_cols(required_cols=df_type, non_empty_row=have_non_empty_row,
                                                               have_na=cols_have_na)))
    test_tdf = test_tdf.convert_datetimes()
    test_tdf = test_tdf.categorize(set_order=True)
    test_tdf = test_tdf.order_categories()
    de_obj = DataExplorer(test_tdf)
    de_obj.filters.show()
    de_obj.filters._initiate_filters()
    de_obj.filters.filter_group.add_filter()
    de_obj.filters.filter_group.combiner.value = combiner.draw(
        sampled_from(de_obj.filters.filter_group.combiner.values))
    de_obj.filters.filter_group.children[1].convert_to_group()
    de_obj.filters.filter_group.children[1].combiner.value = combiner.draw(
        sampled_from(de_obj.filters.filter_group.children[1].combiner.values))
    children = [de_obj.filters.filter_group.children[0],
                de_obj.filters.filter_group.children[1].children[0].children[0],
                de_obj.filters.filter_group.children[1].children[1].children[0]]
    col_list = ['int_col', 'float_col', 'cat_col', 'bool_col',
                'int_col2', 'float_col2', 'cat_col2', 'bool_col2']  # 'dt_col', 'dt_col2' are to be debugged
    selected_col = select_col.draw(lists(sampled_from(col_list), min_size=3, max_size=3, unique=True))
    j = 0
    for child in children:
        child.select_col.value = selected_col[j]  # select_col.draw(sampled_from(child.select_col.values)) #
        child.condition.value = condition.draw(sampled_from(child.condition.values))
        input_value = input_val.draw(sampled_from(sorted(test_tdf[child.select_col.value].dropna().values)))
        if isinstance(child.input_val.value, list):  # for categorical, boolean or <5% unique values columns
            input_value = [input_value]
        elif isinstance(child.input_val.value, tuple):  # for datetime columns (a tuple of 2 values would require)
            input_value2 = input_val.draw(sampled_from(
                test_tdf[test_tdf[child.select_col.value] >= input_value][child.select_col.value].dropna().values))
            input_value = (input_value, input_value2)
        else:  # for numerical columns
            input_value = str(input_value)
        child.input_val.value = input_value
        j += 1
    filter_series = de_obj.filters.get_filters()
    assert isinstance(filter_series.sum(), np.number)


# @pytest.mark.parametrize("cols_have_na",
#                          [True, False],
#                          ids=id_func)
# @pytest.mark.parametrize("have_non_empty_row",
#                          ['have', 'not_have'],
#                          ids=id_func)
# @pytest.mark.parametrize("df_type",
#                          ['Mixed_df2'],
#                          ids=id_func, indirect=True)
# @pytest.mark.parametrize("check_type",
#                          ['C1', 'C2'],
#                          ids=id_func)
# @pytest.mark.parametrize("col_type",
#                          ['num', 'cat', 'date', 'bool'])
# @settings(max_examples=50, deadline=None)
# @given(test_df=data(), col_name=data(), condition=data(), values_input=data(), days_diff=data())
# def test_filter_condition2(test_df, col_name, condition, values_input, df_type, have_non_empty_row, cols_have_na,
#                            check_type, col_type, days_diff):
#     test_tdf = td.DataFrame(test_df.draw(df_with_required_cols(required_cols=df_type, non_empty_row=have_non_empty_row,
#                                                                have_na=cols_have_na)))
#     test_tdf = test_tdf.convert_datetimes()
#     test_tdf = test_tdf.categorize(set_order=True)
#     test_tdf = test_tdf.order_categories()
#     de_obj = DataExplorer(test_tdf)
#     de_obj.create_pane()
#     if check_type == 'C1':
#         de_obj.filters.filter_group.add_new_filter()
#     elif check_type == 'C2':
#         de_obj.filters.filter_group.add_new_filter()
#         de_obj.filters.filter_group.add_new_filter()
#     if not (cols_have_na == True and have_non_empty_row == 'not_have'):
#         for child in de_obj.filters.filter_group.children:
#             if col_type == 'num':
#                 child.col_name.value = col_name.draw(sampled_from(['int_col', 'float_col', 'int_col2', 'float_col2']))
#                 child.condition.value = condition.draw(sampled_from(child.condition.values))
#                 input_val = values_input.draw(sampled_from(test_tdf[child.col_name.value].dropna().unique().tolist()))
#             elif col_type == 'cat':
#                 child.col_name.value = col_name.draw(sampled_from(['cat_col', 'cat_col2']))
#                 child.condition.value = condition.draw(sampled_from(child.condition.values))
#                 input_val = values_input.draw(sampled_from(child.values_input.values))
#             elif col_type == 'date':
#                 child.col_name.value = col_name.draw(sampled_from(['dt_col', 'dt_col2']))
#                 child.condition.value = condition.draw(sampled_from(child.condition.values))
#                 time_diff = (child.values_input.end - child.values_input.start).days
#                 input_val1 = child.values_input.start + timedelta(
#                     days=days_diff.draw(integers(min_value=0, max_value=time_diff)))
#                 input_val2 = child.values_input.start + timedelta(
#                     days=days_diff.draw(integers(min_value=0, max_value=time_diff)))
#                 input_val = [input_val2, input_val1]
#                 input_val.sort()
#             else:
#                 child.col_name.value = col_name.draw(sampled_from(['bool_col', 'bool_col2']))
#                 child.condition.value = condition.draw(sampled_from(child.condition.values))
#                 input_val = values_input.draw(sampled_from(child.values_input.values))
#             if col_type != 'date':
#                 if isinstance(child.values_input.value, list):
#                     input_val = [input_val]
#                 elif isinstance(child.values_input.value, tuple):
#                     input_val = tuple([input_val])
#                 else:
#                     input_val = str(input_val)
#             else:
#                 input_val = tuple(input_val)
#             child.values_input.value = input_val
#         de_obj.filter_data()
#         filter_result = de_obj.filtered_data
#         assert isinstance(filter_result, td.DataFrame)
