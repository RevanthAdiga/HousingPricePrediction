import os
import random
import string
import pandas as pd
import pytest
from tigerml.viz.data_exploration import DataExplorer
from tigerml.viz.widget.components.ys.ui import AGGS
from hypothesis import given, settings
from hypothesis.extra.pandas import columns, data_frames, range_indexes, column
from hypothesis.strategies import composite, dates, floats, integers, booleans, data, sampled_from, lists
import datetime
from tigerml.core.utils.pandas import get_non_num_cols, get_bool_cols, get_cat_cols, get_num_cols
import tigerml.core.dataframe as td


HERE = os.path.dirname(os.path.abspath(__file__))


@composite
def mixed_tup(draw):
    string_options = [
        *[f"Str{str(e)}" for e in list(range(1, 23))],
        *[random.choice(string.ascii_uppercase) + "_caps" for i in range(1, 10)],
        *[random.choice(string.ascii_lowercase) + "_lower" for i in range(1, 10)],
    ]
    float_val = draw(floats(allow_infinity=False, allow_nan=False, min_value=1))
    date_val = draw(dates())
    bool_val = draw(booleans())
    int_val = draw(integers(min_value=1, ))
    str_val = draw(sampled_from(string_options))
    return tuple([float_val, date_val, bool_val, int_val, str_val])


@composite
def mixed_df(draw):
    df = draw(data_frames(
        index=range_indexes(min_size=5, max_size=10),
        columns=columns(
            [
                "float_col",
                "date_col",
                "bool_col",
                "int_col",
                "string_col"
            ],
            dtype=float,
        ),
        rows=mixed_tup(),
    ))
    df.loc[len(df)] = [1.1, datetime.datetime(2018, 10, 8), True, 1, 'A']
    df.loc[len(df)] = [1.2, datetime.datetime(2018, 9, 26), True, 2, 'a']
    df.loc[len(df)] = [1.3, datetime.datetime(2018, 3, 7), False, 3, 'b']
    return df


# Testing the set_values for numeric columns
@pytest.mark.parametrize("col_name",
                         ['int_col', 'float_col'])
@pytest.mark.parametrize("color_axis",
                         ['', 'bool_col', 'date_col', 'string_col'])
@pytest.mark.parametrize("plot_type",
                         ['kde', 'hist', 'box', 'violin', 'table'])  # ['kde', 'hist', 'box', 'violin']
@pytest.mark.parametrize("test_type",
                         ['t1', 't2', 't3'])  # 't4' has to be debugged
@settings(max_examples=10, deadline=None)
@given(test_df=mixed_df(), agg_func=data())
def test_set_values(test_df, col_name, color_axis, plot_type, test_type, agg_func):
    explorer = DataExplorer(test_df)
    explorer.create_pane()
    # simulatge values
    explorer.y_exprs.children[0].col_name.value = col_name
    agg_func_ = agg_func.draw(sampled_from(explorer.y_exprs.children[0].agg_func.values))
    explorer.y_exprs.children[0].agg_func.value = agg_func_
    explorer.y_exprs.children[0].plot_type.value = plot_type
    explorer.color_axis.value = color_axis
    if test_type == 't1':
        assert explorer.y_exprs.children[0].col_name.value == col_name
    elif test_type == 't2':
        assert explorer.y_exprs.children[0].agg_func.value == agg_func_
    elif test_type == 't3':
        assert explorer.color_axis.value == color_axis
    else:
        assert explorer.y_exprs.children[0].plot_type.value == plot_type


# Testing the set_values for non numeric columns
@pytest.mark.parametrize("col_name",
                         ['date_col', 'bool_col', 'string_col'])
@pytest.mark.parametrize("color_axis",
                         ['', 'bool_col', 'date_col', 'string_col'])
@pytest.mark.parametrize("plot_type",
                         ['bar', 'table'])
@pytest.mark.parametrize("test_type",
                         ['t1', 't2', 't4'])  # 't3' has to debugged
@settings(max_examples=10, deadline=None)
@given(test_df=mixed_df(), agg_func=data())
def test_set_values_cont(test_df, col_name, agg_func, color_axis, plot_type, test_type):
    explorer = DataExplorer(test_df)
    explorer.create_pane()
    # simulate values
    explorer.y_exprs.children[0].col_name.value = col_name
    agg_func_ = agg_func.draw(sampled_from(explorer.y_exprs.children[0].agg_func.values))
    explorer.y_exprs.children[0].agg_func.value = agg_func_
    explorer.y_exprs.children[0].plot_type.value = plot_type
    explorer.color_axis.value = color_axis
    if test_type == 't1':
        assert explorer.y_exprs.children[0].col_name.value == col_name
    elif test_type == 't2':
        assert explorer.y_exprs.children[0].agg_func.value == agg_func_
    elif test_type == 't3':
        assert explorer.color_axis.value == color_axis
    else:
        assert explorer.y_exprs.children[0].plot_type.value == plot_type


# Testing the plot generation scenario with y_col
@pytest.mark.parametrize("y_col",
                         ['float_col', 'date_col', 'int_col', 'string_col'])  # 'bool_col' fails in testing but in UI it fails only during StateTracking
@settings(max_examples=50, deadline=None)
@given(test_df=mixed_df(), color_axis=data(), segment_by=data(), agg_f=data(), plot_type=data())
def test_univariate_plot(test_df, y_col, color_axis, segment_by, agg_f, plot_type):
    explorer = DataExplorer(test_df)
    explorer.create_pane()
    # simulate values
    explorer.y_exprs.children[0].col_name.value = y_col
    explorer.color_axis.value = color_axis.draw(sampled_from(explorer.color_axis.values))
    segment_by_val = [True, False]
    explorer.y_exprs.children[0].have_color_axis.value = segment_by.draw(sampled_from(segment_by_val))
    explorer.y_exprs.children[0].agg_func.value = agg_f.draw(sampled_from(explorer.y_exprs.children[0].agg_func.values))
    explorer.y_exprs.children[0].plot_type.value = plot_type.draw(
        sampled_from([x for x in explorer.y_exprs.children[0].plot_type.values if x not in ['table']]))
    explorer.update_plot()
    assert 'holoviews' in str(type(explorer.plot))


# Testing the plot generation scenario with y_col, x_col
@pytest.mark.parametrize("y_col",
                         ['float_col', 'date_col', 'bool_col', 'int_col', 'string_col'])
@pytest.mark.parametrize("x_col",
                         ['', 'float_col', 'date_col', 'bool_col', 'int_col', 'string_col'])
@pytest.mark.parametrize("order",
                         ['with order', 'with out order'])
@settings(max_examples=50, deadline=None)
@given(color_axis=data(), segment_by=data(), agg_f=data(), plot_type=data())
def test_bivariate_plot(y_col, x_col, order, color_axis, segment_by, agg_f, plot_type):
    file_path = os.path.join(HERE, 't1.csv')
    test_df = td.read_csv(file_path)
    test_df = test_df.convert_datetimes()
    if order == 'with order':
        test_df = test_df.categorize(set_order=True)
        test_df = test_df.order_categories()
    explorer = DataExplorer(test_df)
    explorer.create_pane()
    # simulate values
    xy_list = [x_col, y_col].sort()
    str_bool_list = ['string_col', 'bool_col'].sort()
    if x_col != y_col and not (y_col in ['string_col', 'date_col'] and x_col == '') and xy_list != str_bool_list:
        explorer.y_exprs.children[0].col_name.value = y_col
        explorer.x_col.value = x_col
        if not (x_col == 'float_col' and y_col in ['int_col', 'date_col']) and not (
                x_col in get_non_num_cols(test_df) and y_col in get_non_num_cols(test_df)):
            explorer.color_axis.value = color_axis.draw(sampled_from(explorer.color_axis.values))
            segment_by_val = [True, False]
            explorer.y_exprs.children[0].have_color_axis.value = segment_by.draw(sampled_from(segment_by_val))
            if x_col != explorer.y_exprs.children[0].segment_by.value:
                explorer.y_exprs.children[0].agg_func.value = agg_f.draw(
                    sampled_from(explorer.y_exprs.children[0].agg_func.values))
                explorer.y_exprs.children[0].plot_type.value = plot_type.draw(
                    sampled_from([x for x in explorer.y_exprs.children[0].plot_type.values if x not in ['table']]))
                explorer.update_plot()
                assert 'holoviews' in str(type(explorer.plot))


# Testing the plot generation scenario with y_col, x_col, splitter
@pytest.mark.parametrize("y_col",
                         ['float_col', 'date_col', 'bool_col', 'int_col', 'string_col'])
@pytest.mark.parametrize("x_col",
                         ['', 'float_col', 'date_col', 'bool_col', 'int_col', 'string_col'])
@pytest.mark.parametrize("grid_split",
                         [True, False])
@settings(max_examples=20, deadline=None)
@given(splitter=data(), color_axis=data(), segment_by=data(), agg_f=data(), plot_type=data())
def test_bivariate_plot_with_splitter(y_col, x_col, splitter, color_axis, segment_by, agg_f, plot_type, grid_split):
    file_path = os.path.join(HERE, 't1.csv')
    test_df = td.read_csv(file_path)
    test_df = test_df.convert_datetimes()
    test_df = test_df.categorize(set_order=True)
    test_df = test_df.order_categories()
    explorer = DataExplorer(test_df)
    explorer.create_pane()
    # simulate values
    if x_col != y_col and not (y_col in ['string_col', 'date_col'] and x_col == ''):
        explorer.y_exprs.children[0].col_name.value = y_col
        explorer.x_col.value = x_col
        splitter_vals = [x for x in explorer.splitter.values if x not in [x_col, y_col]]
        explorer.splitter.value = splitter.draw(
            lists(elements=sampled_from(splitter_vals), min_size=0,
                  unique=True))
        explorer.split_plots.value = grid_split
        if len(explorer.splitter.value) == 0:
            explorer.split_plots.value = False
        if not (x_col == 'float_col' and y_col in ['int_col', 'date_col']) and \
                not (x_col in get_non_num_cols(test_df) and y_col in get_non_num_cols(test_df)):
            color_axis_vals = [x for x in explorer.color_axis.values if x not in
                               explorer.splitter.value + [x_col]]
            if '' not in color_axis_vals:
                color_axis_vals += ['']
            explorer.color_axis.value = color_axis.draw(sampled_from(color_axis_vals))
            segment_by_val = [True, False]
            if explorer.color_axis.value:
                explorer.y_exprs.children[0].have_color_axis.value = segment_by.draw(sampled_from(segment_by_val))
            else:
                explorer.y_exprs.children[0].have_color_axis.value = False
            explorer.y_exprs.children[0].agg_func.value = \
                agg_f.draw(sampled_from(explorer.y_exprs.children[0].agg_func.values))
            explorer.y_exprs.children[0].plot_type.value = \
                plot_type.draw(sampled_from([x for x in explorer.y_exprs.children[0].plot_type.values if x not in
                                             ['table']]))
            explorer.update_plot()
            assert 'holoviews' in str(type(explorer.plot))


@pytest.mark.parametrize("y_col1",
                         ['float_col', 'cat_col'])
@pytest.mark.parametrize("x_col",
                         ['cat_col2'])
@pytest.mark.parametrize("y_col2",
                         ['float_col3', 'cat_col3'])
@pytest.mark.parametrize("color_col",
                         ['cat_col4'])
@pytest.mark.parametrize("t_type",
                         ['render', 'non_render'])
@settings(max_examples=50, deadline=None)
@given(segment_by1=data(), agg_f1=data(), plot_type1=data(),
       segment_by2=data(), agg_f2=data(), plot_type2=data())
def test_multiple_yexpr(y_col1, x_col, y_col2, color_col, t_type, segment_by1, agg_f1, plot_type1, segment_by2, agg_f2, plot_type2):
    if x_col != y_col1 and x_col != y_col2:
        file_path = os.path.join(HERE, 't3.csv')
        test_df = td.read_csv(file_path)
        test_df = test_df.convert_datetimes()
        test_df = test_df.categorize(set_order=True)
        test_df = test_df.order_categories()
        explorer = DataExplorer(test_df)
        explorer.create_pane()
        # simulate values
        explorer.y_exprs.children[0].col_name.value = y_col1
        explorer.x_col.value = x_col
        explorer.color_axis.value = color_col
        segment_by1_val = [True, False]
        explorer.y_exprs.children[0].have_color_axis.value = segment_by1.draw(sampled_from(segment_by1_val))
        explorer.y_exprs.children[0].agg_func.value = \
            agg_f1.draw(sampled_from(explorer.y_exprs.children[0].agg_func.values))
        explorer.y_exprs.children[0].plot_type.value = \
            plot_type1.draw(sampled_from(explorer.y_exprs.children[0].plot_type.values))
        explorer.y_exprs.add_new_y()
        explorer.y_exprs.children[1].col_name.value = y_col2
        segment_by2_val = [True, False]
        explorer.y_exprs.children[1].have_color_axis.value = segment_by2.draw(sampled_from(segment_by2_val))
        explorer.y_exprs.children[1].agg_func.value = \
            agg_f2.draw(sampled_from(explorer.y_exprs.children[1].agg_func.values))
        explorer.y_exprs.children[1].plot_type.value = \
            plot_type2.draw(sampled_from(explorer.y_exprs.children[1].plot_type.values))
        # if explorer.y_exprs.children[0].plot_type.value != 'heatmap' and explorer.y_exprs.children[1].plot_type.value != 'heatmap':
        explorer.update_plot()
        assert 'holoviews' in str(type(explorer.plot))
        if t_type == 'render':
            import holoviews as hv
            hv.extension('bokeh')
            p = hv.render(explorer.plot, backend='bokeh')


# @composite
# def df_with_required_cols(draw, required_cols='all', non_empty_row='have', have_na=False):
#     if required_cols == 'all':
#         required_cols = ['int_col', 'float_col', 'cat_col', 'dt_col', 'bool_col']
#     date_options = [pd.Timestamp('1/1/1995'), pd.Timestamp('1/1/2005'), pd.Timestamp('1/1/2015')]
#     float_options = floats(allow_infinity=False, allow_nan=False)
#     string_options = ['red', 'Blue', 'GREEN']
#     bool_options = [True, False]
#     if have_na:
#         date_options += [pd.NaT]
#         float_options = floats(allow_infinity=False, allow_nan=True)
#         string_options += ['NA', None]
#         bool_options += [None]
#
#     column_list = []
#     if 'int_col' in required_cols:
#         column_list += [column('int_col', elements=integers())]
#     if 'int_col2' in required_cols:
#         column_list += [column('int_col2', elements=integers())]
#     if 'float_col' in required_cols:
#         # column_list += [column('float_col', elements=floats(allow_infinity=True, allow_nan=True))]
#         column_list += [column('float_col', elements=float_options)]
#     if 'float_col2' in required_cols:
#         column_list += [column('float_col2', elements=float_options)]
#     if 'cat_col' in required_cols:
#         column_list += [column('cat_col', elements=sampled_from(string_options))]
#     if 'cat_col2' in required_cols:
#         column_list += [column('cat_col2', elements=sampled_from(string_options))]
#     if 'dt_col' in required_cols:
#         column_list += [column('dt_col', elements=sampled_from(date_options))]
#     if 'dt_col2' in required_cols:
#         column_list += [column('dt_col2', elements=sampled_from(date_options))]
#     if 'bool_col' in required_cols:
#         column_list += [column('bool_col', elements=sampled_from(bool_options))]
#     if 'bool_col2' in required_cols:
#         column_list += [column('bool_col2', elements=sampled_from(bool_options))]
#
#     df = draw(data_frames(index=range_indexes(min_size=10), columns=column_list))
#     if non_empty_row == 'have':
#         non_empty_row = True
#     else:
#         non_empty_row = False  # draw(sampled_from([True, False]))
#     if non_empty_row:
#         additional_values = {'int_col': [57, 24, 32], 'int_col2': [67, 34, 42],
#                              'float_col': [5.7, 2.4, 3.2], 'float_col2': [6.7, 3.4, 4.2],
#                              'cat_col': ['red', 'Blue', 'GREEN'], 'cat_col2': ['red', 'Blue', 'GREEN'],
#                              'dt_col': [pd.Timestamp('1/1/1965'), pd.Timestamp('1/1/1975'), pd.Timestamp('1/1/1985')],
#                              'dt_col2': [pd.Timestamp('1/1/1965'), pd.Timestamp('1/1/1975'), pd.Timestamp('1/1/1985')],
#                              'bool_col': [True, False, False], 'bool_col2': [True, False, False]}
#
#         for i in range(3):
#             new_row = {}
#             for col in required_cols:
#                 new_row[col] = additional_values[col][i]
#             df = df.append(pd.Series(new_row), ignore_index=True)
#     return df


# Testing the plot generation scenario with multiple yexpr
# @pytest.mark.parametrize("y_col1",
#                          # ['int_col', 'int_col2', 'float_col', 'float_col2', 'cat_col', 'cat_col2'])
#                          ['int_col', 'int_col2', 'float_col', 'float_col2', 'cat_col', 'cat_col2',
#                           'dt_col', 'dt_col2', 'bool_col', 'bool_col2'])
# @pytest.mark.parametrize("x_col",
#                          # ['int_col', 'int_col2', 'float_col', 'float_col2', 'cat_col', 'cat_col2'])
#                          ['int_col', 'int_col2', 'float_col', 'float_col2', 'cat_col', 'cat_col2',
#                           'dt_col', 'dt_col2', 'bool_col', 'bool_col2'])
# @pytest.mark.parametrize("y_col2",
#                          # ['int_col', 'int_col2', 'float_col', 'float_col2', 'cat_col', 'cat_col2'])
#                          ['int_col', 'int_col2', 'float_col', 'float_col2', 'cat_col', 'cat_col2',
#                           'dt_col', 'dt_col2', 'bool_col', 'bool_col2'])
# @settings(max_examples=10, deadline=None)
# @given(segment_by1=data(), agg_f1=data(), plot_type1=data(),
#        segment_by2=data(), agg_f2=data(), plot_type2=data())
# def test_multiple_yexpr(y_col1, x_col, y_col2, segment_by1, agg_f1, plot_type1, segment_by2, agg_f2, plot_type2):
#     if x_col != y_col1 and x_col != y_col2:
#         file_path = os.path.join(HERE, 't2.csv')
#         test_df = td.read_csv(file_path)
#         test_df = test_df.convert_datetimes()
#         test_df = test_df.categorize(set_order=True)
#         test_df = test_df.order_categories()
#         explorer = DataExplorer(test_df)
#         explorer.create_pane()
#         # simulate values
#         explorer.y_exprs.children[0].col_name.value = y_col1
#         explorer.x_col.value = x_col
#         segment_by1_val = [x for x in explorer.y_exprs.children[0].segment_by.values if x not in [x_col, y_col1]]
#         explorer.y_exprs.children[0].segment_by.value = segment_by1.draw(sampled_from(segment_by1_val))
#         explorer.y_exprs.children[0].agg_func.value = \
#             agg_f1.draw(sampled_from(explorer.y_exprs.children[0].agg_func.values))
#         explorer.y_exprs.children[0].plot_type.value = \
#             plot_type1.draw(sampled_from(explorer.y_exprs.children[0].plot_type.values))
#         explorer.y_exprs.add_new_y()
#         explorer.y_exprs.children[1].col_name.value = y_col2
#         segment_by2_val = [x for x in explorer.y_exprs.children[1].segment_by.values if x not in [x_col, y_col2]]
#         explorer.y_exprs.children[1].segment_by.value = segment_by2.draw(sampled_from(segment_by2_val))
#         explorer.y_exprs.children[1].agg_func.value = \
#             agg_f2.draw(sampled_from(explorer.y_exprs.children[1].agg_func.values))
#         explorer.y_exprs.children[1].plot_type.value = \
#             plot_type2.draw(sampled_from(explorer.y_exprs.children[1].plot_type.values))
#         explorer.update_plot()
#         assert 'holoviews' in str(type(explorer.plot))
