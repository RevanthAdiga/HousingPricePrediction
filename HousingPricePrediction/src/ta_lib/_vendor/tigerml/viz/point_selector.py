import os
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
LASSO_SELECT_LIMIT = 10000  # Lasso select is inefficient, when a lot of points exists, it becomes buggy
PREVIEW_LIMIT = 100000 # Data Preview is limited to these many rows
from .widget.states import StatefulUI


class Selection(StatefulUI):

    def __init__(self, selection_dict, parent, initial_state={}):
        super().__init__(initial_state=initial_state)
        self.parent = parent
        self.name = selection_dict['name']
        self.description = selection_dict['description']
        self.data = selection_dict['data']
        self.expanded = self.showing_data = False
        self._create_ui()

    def _create_ui(self):
        actions = self.togglegroup(options=['include', 'exclude', 'limit to', 'compare'], name='Plot Actions',
                                   width=300, behavior='radio', css_classes=['tabs'],
                                   callback=self.parent.parent.update_plot)
        ui_config = {'actions': actions}
        self._create_ui_from_config(ui_config)

    def show_data(self, event=None):
        data = self.parent.parent.data.loc[self.data]
        self.parent.show_data_preview_overlay(data)

    def get_selection_actions(self):
        merge_btn = self.Button(css_classes=['icon-button', 'icon-merge'], width=30)
        merge_btn.on_click(self.show_options_to_merge)
        compliment_btn = self.Button(css_classes=['icon-button', 'icon-compliment'], width=30)
        download_btn = self.Button(css_classes=['icon-button', 'icon-download'], width=30)
        download_btn.on_click(self.save)
        preview_btn = self.Button(css_classes=['icon-button', 'icon-preview'], width=30)
        preview_btn.on_click(self.show_data)
        delete_btn = self.Button(css_classes=['icon-button', 'icon-delete'], width=30)
        delete_btn.on_click(self.delete_self)
        selection_actions = self.Row(self.HTML('<h3 class="section_header">Actions: </h3>'),
                                     merge_btn, compliment_btn, preview_btn, download_btn, delete_btn)
        ctas = self.Column(selection_actions)
        description = self.Column('<b>Description: </b>' + self.description)
        # show_data_btn = self.Button(name='Show Data')
        self.sel_actions = self.Column(ctas, description)
        return self.sel_actions

    def show_options_to_merge(self, event=None):
        if len(self.parent.children) > 1:
            self.parent.show_options_to_merge(self.data, exclude_sel=self)
        else:
            pass

    def delete_self(self, event=None):
        self.parent.delete_child(self)
        self.parent.refresh_ui()
        self.parent.parent.update_plot()

    def save(self, event=None):
        data = pd.Series([0] * len(self.parent.parent.data), index=self.parent.parent.data.index)
        data.name = self.name
        data.loc[self.data] = 1
        data.to_csv(f'{self.name}_flags.csv')
        del data

    def get_ui(self):
        selection_link_template = '''
            <div class="selection">
                <a>{}</a>
                <div class="icon-link icon-edit"></div>
            </div>
        '''
        self.pane = self.Column(self.Column(
            self.HTML(selection_link_template.format(self.name)),
            self.Row(self.actions, '<div class="icon-link icon-expand"></div>')))
        if self.expanded:
            self.pane.css_classes = ['expanded']
            self.pane.append(self.get_selection_actions())
        return self.pane


class Selections(StatefulUI):

    def __init__(self, parent, initial_state={}):
        self.parent = parent
        super().__init__(children=[], dynamic_ui=True, initial_state=initial_state)

    def save_selection(self, sel_id):
        selection = self.children[sel_id]
        selection.save()

    def refresh_ui(self):
        # self.selection_pane[3] = self.get_selections()
        self.refresh_selections()

    def _add_child(self, selection_dict):
        selection = Selection(selection_dict, self)
        self.children.append(selection)
        self.save_current_state()

    def get_selections(self):
        self.selections_ui = self.Column(css_classes=['selections'])
        for selection in self.children:
            self.selections_ui.append(selection.get_ui())
        return self.selections_ui

    def show_selection_actions(self, event=None):
        selection_index = self.sel_select.value
        selection = self.children[selection_index]
        selection.expanded = not selection.expanded     # Toggle the value of expanded
        self.refresh_ui()

    def get_ui(self):
        self.pane = self.Column(css_classes=['selection_widget'])
        self.sel_select = self.Select(css_classes=['selection_input'])
        self.sel_name_edit = self.Button(css_classes=['edit_selection'], width=0)
        self.sel_expand = self.Button(css_classes=['expand_selection'], width=0)

        self.sel_name_input = self.TextInput(css_classes=['selection_name'], width=240)
        self.sel_name_save = self.Button(css_classes=['icon-button', 'icon-ok'], width=30)
        self.cancel = self.Button(css_classes=['icon-button', 'icon-cancel'], width=30)
        self.selection_edit = self.Row(self.sel_name_input, self.sel_name_save, self.cancel,
                                       css_classes=['content'])
        edit_widget = self.Column(self.HTML('<div class="glass"></div>'), self.selection_edit,
                                  css_classes=['edit_selection_widget', 'overlay', 'is_hidden'])
        self.hidden_sel_triggers = self.Row(self.sel_select, self.sel_name_edit, self.sel_expand,
                                            css_classes=['is_hidden'])
        self.sel_expand.on_click(self.show_selection_actions)
        self.sel_name_edit.on_click(self.edit_selection)
        self.pane.extend([self.hidden_sel_triggers, edit_widget])

        self.pane.append(self.get_selections())
        self.get_script()
        self.overlays = self.Column()
        self.pane.extend([self.script, self.overlays])
        return self.pane

    def edit_selection(self, event=None):
        selection_index = self.sel_select.value
        selection = self.children[selection_index]
        selection.name = self.sel_name_input.value
        self.refresh_selections()

    def compute_merge_stats(self, event=None):
        self.summaries[0] = 'Computing...'
        self.summaries[1] = 'Computing...'
        self.merger_selection = [sel for sel in self.children if sel.name == self.merge_options.value][0]
        self.sel_summary = f'Selection: {len(self.merger_selection.data)} points'
        if self.merge_logic.value == 'OR':
            self.merge_result = list(set(self.merger_selection.data) | set(self.merging_data))
        else:
            self.merge_result = list(set(self.merger_selection.data) & set(self.merging_data))
        self.merge_summary = f'After merging: {len(self.merge_result)} points'
        self.summaries[0] = self.sel_summary
        self.summaries[1] = self.merge_summary

    def close_merging(self, event=None):
        merge_ui = self.overlays.pop(-1)
        del merge_ui

    def init_merging(self, sel_options=None):
        sel_options = sel_options if sel_options is not None else [sel.name for sel in self.children]
        self.merge_options = self.Select(name='Select Selection to Merge', options=sel_options)
        self.merge_logic = self.Select(name='Merge Logic', options=['OR', 'AND'])
        self.data_summary = self.Row(f'Data: {len(self.merging_data)} points')
        self.sel_summary = ''
        self.merge_summary = ''
        self.merge_options.param.watch(self.compute_merge_stats, 'value')
        self.merge_logic.param.watch(self.compute_merge_stats, 'value')
        close_btn = self.Button(css_classes=['icon-button', 'icon-cancel'], width=50)
        close_btn.on_click(self.close_merging)
        self.merge_btn = self.Button(name='Merge Datasets')
        self.merge_btn.on_click(self.merge_with_selection)
        self.summaries = self.Column(self.sel_summary, self.merge_summary)
        self.merge_overlay = self.Column(
            close_btn,
            self.data_summary,
            self.Row(self.merge_options, self.merge_logic),
            self.summaries,
            self.merge_btn,
            css_classes=['content', 'medium'])
        self.compute_merge_stats()

    def merge_with_selection(self, event=None):
        self.merger_selection.data = self.merge_result
        self.merger_selection.name = f'Merged Selection of {len(self.merger_selection.data)} points'
        self.close_merging()
        self.refresh_selections()

    def show_options_to_merge(self, data, exclude_sel=None):
        self.merging_data = data
        if exclude_sel:
            sel_options = [sel.name for sel in self.children if sel != exclude_sel]
        else:
            sel_options = [sel.name for sel in self.children]
        if not hasattr(self, 'merge_overlay'):
            self.init_merging(sel_options=sel_options)
        else:
            self.data_summary[0] = f'Data: {len(self.merging_data)} points'
            self.merge_options.options = sel_options
        self.overlays.append(self.Column(self.HTML('<div class="glass"></div>'), self.merge_overlay,
                                         css_classes=['overlay']))

    def hide_data(self, event=None):
        data_ui = self.overlays.pop(-1)
        del data_ui

    def show_data_preview_overlay(self, data):
        hide_data_btn = self.Button(css_classes=['icon-button', 'icon-cancel'], width=50)
        hide_data_btn.on_click(self.hide_data)
        from tigerml.core.plots import hvPlot
        message = ''
        if len(data) > PREVIEW_LIMIT:
            message = f'Data is too big. Limiting preview to {PREVIEW_LIMIT} row'
            data = data[:PREVIEW_LIMIT]
        data_preview = self.Column(self.HTML('<div class="glass"></div>'),
                                   self.Column(hide_data_btn, message, hvPlot(data).table(
                                       width=(len(data.columns) + 1) * 150), css_classes=['content', 'big']),
                                   css_classes=['data_preview', 'overlay'], width=800)
        self.overlays.append(data_preview)

    def get_script(self):
        script_path = f'{HERE}/static_resources/selection_script.js'
        script_file = open(script_path, "r")
        custom_script = script_file.read()
        script_file.close()
        script = '<script>{}</script>'.format(custom_script)
        self.script = self.HTML(script)

    def refresh_selections(self):
        self.sel_select.options = list(range(0, len(self.children)))
        self.pane[3] = self.get_selections()
        self.pane[4] = self.script


class PointSelector(Selections):

    def __init__(self, parent, initial_state={}):
        self.reset_selection()
        self.sel_type = 'xy'
        super().__init__(parent, initial_state=initial_state)

    def reset_selection(self):
        self.y_expr = None
        self.selected_points = None
        self.selected_data = None
        self.current_y = None
        self.current_x = None
        self.y_selection = self.x_selection = None

    @property
    def selection_description(self):
        if self.sel_type == 'xy':
            x_desc = f'{self.x_selection[0]} < {self.current_x} < {self.x_selection[1]}' \
                if isinstance(self.x_selection, tuple) else f'{self.current_x} in {self.selected_points.index.unique().tolist()}' \
                if self.current_x and self.x_selection else ''
            y_desc = f'{self.y_selection[0]} < {self.current_y} < {self.y_selection[1]}' \
                if isinstance(self.y_selection, tuple) else f'{self.current_y} in {self.selected_points[self.current_y].unique().tolist()}' \
                if self.current_y and self.y_selection else ''
            if x_desc and y_desc:
                desc = ' and '.join([x_desc, y_desc])
            else:
                desc = x_desc or y_desc
            name = f'{desc} ({len(self.selected_data)} points)'
        else:
            name = f'Selected points of {self.current_y} vs {self.current_x} ({len(self.selected_data)})'
        return name

    def create_selection(self, event=None):
        # import pdb
        # pdb.set_trace()
        name = f'Selection of {len(self.selected_data)} points'
        description = self.selection_description
        self._add_child({'data': self.selected_data, 'name': name, 'description': description})
        self.refresh_ui()

    def set_selection_tools(self, y_expr, current_plot, plot_kwargs, x_col, plotter=None):
        self.reset_selection()
        self.current_y = y_expr.display_name
        self.current_x = x_col
        self.y_expr = y_expr
        self.widget = y_expr.parent.parent
        self.sel_type = 'xy'
        if y_expr.plot_type.value == 'table':
            y_expr.selection_summary = ''
            return current_plot
        from holoviews import streams
        import holoviews as hv
        from tigerml.core.plots import hvPlot
        import pandas as pd

        def get_original_data(filtered_data):
            original_filter = None
            if x_col:
                original_filter = (y_expr.source_data[filtered_data.index.name].isin(filtered_data.index))
            if original_filter is None:
                original_filter = (y_expr.source_data[y_expr.display_name].isin(filtered_data[y_expr.display_name]))
            else:
                original_filter &= (y_expr.source_data[y_expr.display_name].isin(filtered_data[y_expr.display_name]))
            original_data = y_expr.source_data[original_filter].index.values.tolist()
            return original_data

        def compute_summary_from_index(index):
            if index:
                self.selected_points = y_expr.plot_data.iloc[index]
                self.selected_data = get_original_data(self.selected_points)
                self.selection_summary_ui[0] = self.selected_summary_ui()
                selection_summary = pd.DataFrame({'selected_points': len(self.selected_points),
                                                  'original_points': len(self.selected_data)}, index=[0])
                print(selection_summary)
                return hvPlot(selection_summary).table()
            else:
                self.selected_points = None
                self.selected_data = None
                self.selection_summary_ui[0] = self.selected_summary_ui()
                return hvPlot(pd.DataFrame()).table()

        def compute_summary_from_box(y_selection, x_selection, bounds):
            self.x_selection = x_selection
            self.y_selection = y_selection
            if y_selection is None and x_selection is None:
                self.selected_points = None
                self.selected_data = None
                self.selection_summary_ui[0] = self.selected_summary_ui()
                return hvPlot(pd.DataFrame()).table()
            else:
                x_filter = y_filter = None
                if x_col or y_expr.plot_type.value in ['hist', 'kde']:
                    if isinstance(x_selection, tuple):
                        if y_expr.mapper_df is not None:
                            mapper_filter = (x_selection[0] < y_expr.mapper_df['tigerml_mapper']) & \
                                            (y_expr.mapper_df['tigerml_mapper'] < x_selection[1])
                            x_filter = y_expr.plot_data.index.isin(y_expr.mapper_df[mapper_filter][x_col])
                        else:
                            x_filter = ((y_expr.plot_data.index >= x_selection[0]) & (y_expr.plot_data.index <= x_selection[1]))
                    else:
                        x_filter = (y_expr.plot_data.index.isin(x_selection))
                if y_expr.plot_type.value not in ['hist', 'kde']:
                    if isinstance(y_selection, tuple):
                        y_filter = ((y_expr.plot_data[y_expr.display_name] >= y_selection[0]) &
                                    (y_expr.plot_data[y_expr.display_name] <= y_selection[1])).values
                    else:
                        y_filter = (y_expr.plot_data[y_expr.display_name].isin(y_selection)).values
                final_filter = None
                if x_filter is not None:
                    final_filter = x_filter
                if y_filter is not None:
                    if final_filter is None:
                        final_filter = y_filter
                    else:
                        final_filter &= y_filter
                self.selected_points = y_expr.plot_data[final_filter]
                self.selected_data = get_original_data(self.selected_points)
                self.selection_summary_ui[0] = self.selected_summary_ui()
                selection_summary = pd.DataFrame({'selected_points': len(self.selected_points),
                                                  'original_points': len(self.selected_data)}, index=[0])
                print(selection_summary)
                return hvPlot(selection_summary).table()

        if (y_expr.segment_by and y_expr.plot_type.value not in ['box', 'kde', 'grouped_bar']) or \
                y_expr.plot_type.value == 'bar':
            data = pd.DataFrame.from_dict({'points': [y_expr.plot_data[y_expr.display_name].min(),
                                                      y_expr.plot_data[y_expr.display_name].max()]})
            data.index = [y_expr.plot_data.index.values[0], y_expr.plot_data.index.values[-1]]
            scatter_kwargs = plot_kwargs.copy()
            scatter_kwargs['kind'] = 'scatter'
            scatter_kwargs['s'] = 0
            dummy_scatter = plotter(**scatter_kwargs)  # hvPlot(data).scatter(s=0)
            entire_dummy_scatter = dummy_scatter
            if dummy_scatter.__module__ == 'panel.layout':  # If the plot is split, it will be a panel layout on which
                # the further processing won't work. Need to extract the plot inside the layout to post process.
                dummy_scatter = dummy_scatter[1].object
            dummy_scatter.opts(tools=['box_select', 'hover'])
            selector = streams.SelectionXY(source=dummy_scatter)
            y_expr.selection_summary = hv.DynamicMap(compute_summary_from_box, kdims=[], streams=[selector])
            current_plot = current_plot * dummy_scatter
        else:
            if y_expr.plot_type.value == 'scatter' and not y_expr.parent.parent.splitter.value and \
                    'datashade' not in plot_kwargs and len(y_expr.plot_data) < LASSO_SELECT_LIMIT:
                print('all scatter plot')
                current_plot.opts(tools=['box_select', 'lasso_select', 'hover'])
                self.sel_type = 'index'
                selector = streams.Selection1D(source=current_plot)
                y_expr.selection_summary = hv.DynamicMap(compute_summary_from_index, kdims=[], streams=[selector])
            else:
                current_plot.opts(tools=['box_select', 'hover'])
                selector = streams.SelectionXY(source=current_plot)
                y_expr.selection_summary = hv.DynamicMap(compute_summary_from_box, kdims=[], streams=[selector])
        y_expr.selection_summary.opts(width=300, height=50)
        return current_plot

    def add_selected_points_plot(self, y_expr, current_plot, plot_kwargs):
        return current_plot

    def show_selections_to_merge(self, event=None):
        self.show_options_to_merge(self.selected_data)

    def show_selected_points(self, event=None):
        data = self.selected_points
        self.show_data_preview_overlay(data)

    def show_selected_data(self, event=None):
        data = self.parent.data.loc[self.selected_data]
        self.show_data_preview_overlay(data)

    def selected_summary_ui(self):
        if self.y_expr:
            hide = False
            if self.selected_points is None:
                hide = True
            if hide:
                self.selection_summary = self.Column('<b>Selected Points</b>', css_classes=['is_hidden'], width=0, height=0)
            else:
                self.selection_summary = self.Column('<b>Selected Points</b>', css_classes=[])
            self.selection_summary.append(self.y_expr.selection_summary)
            self.add_selection_btn = self.Button(name='Save', css_classes=['secondary', 'button'], width=100)
            self.add_selection_btn.on_click(self.create_selection)
            self.ctas = self.Row(self.add_selection_btn)
            self.merge_selection_btn = self.Button(name='Merge', css_classes=['tertiary', 'button'], width=100)
            self.merge_selection_btn.on_click(self.show_selections_to_merge)
            if self.children:
                self.ctas.append(self.merge_selection_btn)
            self.show_points_btn = self.Button(name='Show Selected Points', width=150)
            self.show_points_btn.on_click(self.show_selected_points)
            self.show_data_btn = self.Button(name='Show Selected Data', width=150)
            self.show_data_btn.on_click(self.show_selected_data)
            self.preview_options = self.Row(self.show_points_btn, self.show_data_btn)
            self.selection_summary.extend([self.preview_options, self.ctas])
            return self.selection_summary
        else:
            return self.null_component

    def get_selection_summary(self):
        self.selection_summary_ui = self.Column(self.selected_summary_ui())
        return self.selection_summary_ui

    def get_ui(self):
        super().get_ui()
        self.pane.insert(0, self.get_selection_summary())
        return self.pane

