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


@settings(max_examples=10, deadline=None)
@given(test_df=mixed_df())
def test_initial_values(test_df):
    explorer = DataExplorer(test_df)
    # assert explorer.y_exprs.children[0].col_name.value == list(test_df.columns)[0]
    assert explorer.y_exprs.children[0].agg_func.value == ""
    assert explorer.y_exprs.children[0].sort_rule.value == ""
    assert explorer.y_exprs.children[0].axis.value == "left"
    assert explorer.y_exprs.children[0].plot_type.value is None
    assert explorer.y_exprs.children[0].segment_by == ""
    assert all(explorer.y_exprs.children[0].col_name.values) == all(test_df.columns)
    assert all(explorer.y_exprs.children[0].agg_func.values) == all(list(AGGS.keys()))
    assert all(explorer.y_exprs.children[0].sort_rule.values) == all(["", "ASC", "DESC"])
    assert all(explorer.y_exprs.children[0].axis.values) == all(["left", "right"])
    assert explorer.y_exprs.children[0].plot_type.values == []
    initial_segments = ["", "date_col1", "bool_col1", "string_col1"]
    explorer.create_pane()
    assert all(explorer.color_axis.values) == all(initial_segments)


@pytest.mark.parametrize("col_name, num_col",
                         [("float_col", True),
                          ("date_col", False),
                          ("int_col", True),
                          ("string_col", False)])  # ("bool_col", False) has to be debugged
@settings(max_examples=10, deadline=None)
@given(test_df=mixed_df())
def test_options(test_df, col_name, num_col):
    explorer = DataExplorer(test_df)
    explorer.create_pane()
    # simulate values
    explorer.y_exprs.children[0].col_name.value = col_name
    expected_segmentby_options = [""] + get_non_num_cols(test_df)
    if num_col:
        assert all(explorer.y_exprs.children[0].agg_func.options) == \
               all(['', 'sum', 'mean', 'max', 'min', 'count', 'distinct count'])
        assert all(explorer.color_axis.options) == all(expected_segmentby_options)
        assert all(explorer.y_exprs.children[0].plot_type.options) == all(['kde', 'hist', 'box', 'violin', 'table'])
    else:
        assert all(explorer.y_exprs.children[0].agg_func.options) == all(['', 'count', 'distinct count'])
        expected_segmentby_options.remove(col_name)
        assert all(explorer.color_axis.options) == all(expected_segmentby_options)
        assert all(explorer.y_exprs.children[0].plot_type.options) == all(['bar', 'table'])
