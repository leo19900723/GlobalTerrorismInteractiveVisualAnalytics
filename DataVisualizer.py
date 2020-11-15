import json

import pandas as pd
import dash
import dash_html_components as html
import plotly.express as px
import dash_core_components as dcc
import plotly.graph_objects as go
import calendar

from plotly.subplots import make_subplots
from DataHandler import DataHandler
from colour import Color


class DataVisualizer(object):

    def __init__(self, data_handler, app, mapbox_token, mapbox_style):
        self._data_handler = data_handler
        self._app = app

        self._default_web_title = "Project 2: Interactive Visual Analytics Dashboard"
        self._default_web_credit = "Yi-Chen Liu © 2020 Copyright held by the owner/author(s)."

        self._default_year_range = [1994, 2017]
        self._default_number_of_reserved = 8
        self._default_column_pick = self._data_handler.get_txt_columns[0]
        self._default_pca_target_pick = "region_txt"

        self._default_plain_fig = dict(
            data=[dict(x=0, y=0)],
            layout=dict(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=20, r=20, b=20, l=20),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False)
            )
        )

        self._default_colors = {
            "screen1": {"light": "#C4DBFF", "dark": "#202E45"},
            "screen2": {"light": "#C1FFFA", "dark": "#10523E"},
            "screen3": {"light": "#FDDFD5", "dark": "#3A0000"}
        }

        self._pca_clustering_df = {}
        self._pca_clustering_target = None
        self._pca_calc_trial_num = 1

        self._mapbox_token = mapbox_token
        self._mapbox_style = mapbox_style

        self.set_layout()
        self.callback()

    @classmethod
    def construct_from_csv(cls, path):
        external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
        mapbox_info_file_path = "assets/mapbox_info.json"

        with open(mapbox_info_file_path) as mapbox_info_file:
            mapbox_info_dict = json.load(mapbox_info_file)

        return DataVisualizer(
            data_handler=DataHandler.construct_from_csv(path),
            app=dash.Dash(__name__, external_stylesheets=external_stylesheets),
            mapbox_token=mapbox_info_dict["mapbox_token"],
            mapbox_style=mapbox_info_dict["mapbox_style"]["UCDavis_289H_Project2_Dark"]
        )

    def set_layout(self):
        self._app.title = self._default_web_title
        self._app.layout = html.Div(id="main", children=[
            html.Div(
                id="side_bar",
                children=[
                    html.Div(
                        id="side_bar_top",
                        children=[
                            html.H1(children=self._default_web_title),
                            html.Div(children=self._default_web_credit)
                        ]
                    ),

                    html.Div(
                        id="side_bar_bottom",
                        children=[
                            html.Div(
                                id="side_bar_bottom0",
                                children=[
                                    html.H5(children="Year Range Picker"),
                                    html.Div(id="year_slider_frame", children=[
                                        dcc.RangeSlider(
                                            id="year_slider",
                                            min=self._data_handler.get_data_frame_original["iyear"].min(),
                                            max=self._data_handler.get_data_frame_original["iyear"].max(),
                                            value=self._default_year_range
                                        )
                                    ]),

                                    html.H5(children="Specific Year Picker"),
                                    dcc.Dropdown(
                                        id="specific_year_picker",
                                        placeholder="Select a year"
                                    ),

                                    html.H5(children="Column Picker"),
                                    dcc.Dropdown(
                                        id="column_picker",
                                        options=[{"label": "View: " + col, "value": col} for col in
                                                 self._data_handler.get_txt_columns],
                                        value=self._default_column_pick,
                                        clearable=False,
                                        placeholder="Select a column in the dataset"
                                    ),

                                    html.H5(children="Detail Categories Picker"),
                                    dcc.Dropdown(id="categories_picker", multi=True)
                                ]
                            ),

                            html.Div(
                                id="side_bar_bottom1",
                                children=[
                                    html.H5(children="PCA Target Picker"),
                                    dcc.Dropdown(
                                        id="ml_target_picker",
                                        options=[{"label": col, "value": col} for col in
                                                 self._data_handler.get_data_frame_original.columns],
                                        clearable=False,
                                        value=self._default_pca_target_pick
                                    ),

                                    html.H5(children="PCA/ K-Means Feature Columns Picker"),
                                    dcc.Dropdown(
                                        id="ml_feature_cols_picker",
                                        options=[{"label": col, "value": col} for col in
                                                 self._data_handler.get_numeric_columns],
                                        value=self._data_handler.get_numeric_columns,
                                        multi=True
                                    ),

                                    html.Div(id="side_bar_bottom1_parameters", children=[
                                        html.Div(id="ml_num_of_pc_frame", children=[
                                            html.H6(children="Principle Columns"),
                                            dcc.Input(
                                                id="ml_num_of_pc_setup",
                                                type="number",
                                                min=3,
                                                value=6
                                            )
                                        ]),

                                        html.Div(id="ml_random_state_frame", children=[
                                            html.H6(children="Random State"),
                                            dcc.Input(
                                                id="ml_random_state_setup",
                                                type="number",
                                                value=5
                                            )
                                        ]),
                                    ]),

                                    html.Button(id="ml_calc_button_pca_var",
                                                children=[html.Span("Calculate")],
                                                n_clicks=self._pca_calc_trial_num),

                                    html.Div(id="ml_axis_picker", children=[
                                        html.Div(id="ml_axis_picker_x_frame", children=[
                                            html.H6(children="X-Axis for 3D"),
                                            dcc.Dropdown(id="ml_axis_picker_x",
                                                         clearable=False,
                                                         placeholder="Select X-Axis")
                                        ]),

                                        html.Div(id="ml_axis_picker_y_frame", children=[
                                            html.H6(children="Y-Axis for 3D"),
                                            dcc.Dropdown(id="ml_axis_picker_y",
                                                         clearable=False,
                                                         placeholder="Select Y-Axis")
                                        ]),

                                        html.Div(id="ml_axis_picker_z_frame", children=[
                                            html.H6(children="Z-Axis for 3D"),
                                            dcc.Dropdown(id="ml_axis_picker_z",
                                                         clearable=False,
                                                         placeholder="Select Z-Axis")
                                        ])
                                    ])
                                ]
                            )
                        ]
                    ),
                ]
            ),

            html.Div(
                id="content",
                children=[
                    html.Div(
                        id="screen0",
                        children=[
                            html.Div(
                                id="screen00",
                                children=[
                                    dcc.Graph(id="bar_year_attack_type_all_fig",
                                              figure=self._default_plain_fig,
                                              className="graph_style")
                                ]
                            ),

                            html.Div(
                                id="screen01",
                                children=[
                                    dcc.Graph(id="heatmap_weekday_month_year_attack_freq_fig",
                                              figure=self._default_plain_fig,
                                              className="graph_style")
                                ]
                            )
                        ]
                    ),

                    html.Div(
                        id="screen1",
                        children=[
                            html.Div(
                                id="screen10",
                                children=[
                                    dcc.Graph(id="map_year_attack_type_all_or_specified_fig",
                                              figure=self._default_plain_fig,
                                              className="graph_style")
                                ]
                            ),

                            html.Div(
                                id="screen11",
                                children=[
                                    dcc.Graph(id="pies_kill_wound_nationality_selected_points",
                                              figure=self._default_plain_fig,
                                              className="graph_style")
                                ]
                            )
                        ]
                    ),

                    html.Div(
                        id="screen2",
                        children=[
                            html.Div(
                                id="screen20",
                                children=[
                                    html.Div(id="screen200", children=[
                                        dcc.Loading(children=[
                                            dcc.Graph(id="pca_2d_matrix",
                                                      figure=self._default_plain_fig, className="graph_style")
                                        ])
                                    ]),
                                    html.Div(id="screen201", children=[
                                        dcc.Loading(children=[
                                            dcc.Graph(id="clustering_2d_matrix",
                                                      figure=self._default_plain_fig, className="graph_style")
                                        ])
                                    ])
                                ]
                            ),
                            html.Div(
                                id="screen21",
                                children=[
                                    html.Div(id="screen210", children=[
                                        html.Div(id="screen2000", children=[
                                            dcc.Loading(children=[
                                                dcc.Graph(id="pca_3d",
                                                          figure=self._default_plain_fig, className="graph_style")
                                            ])
                                        ]),
                                        html.Div(id="screen2001", children=[
                                            dcc.Loading(children=[
                                                dcc.Graph(id="clustering_3d",
                                                          figure=self._default_plain_fig, className="graph_style")
                                            ])
                                        ])
                                    ]),
                                    html.Div(id="screen211", children=[
                                        dcc.Graph(id="heatmap_correlation_selected_cols",
                                                  figure=self._default_plain_fig, className="graph_style")
                                    ])
                                ]
                            )
                        ]
                    )
                ]
            )
        ])

    def callback(self):

        # Callback for the control panel
        self._app.callback(
            dash.dependencies.Output("categories_picker", "options"),
            dash.dependencies.Output("categories_picker", "value"),
            [dash.dependencies.Input("column_picker", "value")]
        )(self._update_categories_picker)

        self._app.callback(
            dash.dependencies.Output("specific_year_picker", "value"),
            dash.dependencies.Output("specific_year_picker", "options"),
            [dash.dependencies.Input("year_slider", "value"),
             dash.dependencies.Input("bar_year_attack_type_all_fig", "selectedData")]
        )(DataVisualizer._update_specific_year_picker)

        self._app.callback(
            dash.dependencies.Output("ml_axis_picker_x", "options"),
            dash.dependencies.Output("ml_axis_picker_x", "value"),
            dash.dependencies.Output("ml_axis_picker_y", "options"),
            dash.dependencies.Output("ml_axis_picker_y", "value"),
            dash.dependencies.Output("ml_axis_picker_z", "options"),
            dash.dependencies.Output("ml_axis_picker_z", "value"),
            [dash.dependencies.Input("ml_num_of_pc_setup", "value")]
        )(DataVisualizer._update_ml_axis_picker)

        # Callback for graphics
        self._app.callback(
            dash.dependencies.Output("bar_year_attack_type_all_fig", "figure"),
            [dash.dependencies.Input("year_slider", "value")]
        )(self._update_bar_year_attack_type_all_fig)

        self._app.callback(
            dash.dependencies.Output("heatmap_weekday_month_year_attack_freq_fig", "figure"),
            [dash.dependencies.Input("year_slider", "value"),
             dash.dependencies.Input("specific_year_picker", "value")]
        )(self._update_heatmap_weekday_month_year_attack_freq_fig)

        self._app.callback(
            dash.dependencies.Output("map_year_attack_type_all_or_specified_fig", "figure"),
            [dash.dependencies.Input("year_slider", "value"),
             dash.dependencies.Input("specific_year_picker", "value"),
             dash.dependencies.Input("column_picker", "value"),
             dash.dependencies.Input("categories_picker", "value")]
        )(self._update_map_year_attack_type_all_or_specified_fig)

        self._app.callback(
            dash.dependencies.Output("pies_kill_wound_nationality_selected_points", "figure"),
            [dash.dependencies.Input("map_year_attack_type_all_or_specified_fig", "selectedData"),
             dash.dependencies.Input("year_slider", "value"),
             dash.dependencies.Input("specific_year_picker", "value"),
             dash.dependencies.Input("column_picker", "value"),
             dash.dependencies.Input("categories_picker", "value")]
        )(self._update_pies_kill_wound_nationality_selected_points)

        self._app.callback(
            dash.dependencies.Output("pca_2d_matrix", "figure"),
            dash.dependencies.Output("clustering_2d_matrix", "figure"),
            dash.dependencies.Output("ml_calc_button_pca_var", "children"),
            [dash.dependencies.Input("ml_calc_button_pca_var", "n_clicks")],
            [dash.dependencies.State("ml_target_picker", "value"),
             dash.dependencies.State("ml_feature_cols_picker", "value"),
             dash.dependencies.State("ml_num_of_pc_setup", "value"),
             dash.dependencies.State("ml_random_state_setup", "value")]
        )(self._update_pca_clustering_matrix)

        self._app.callback(
            dash.dependencies.Output("pca_3d", "figure"),
            dash.dependencies.Output("clustering_3d", "figure"),
            [dash.dependencies.Input("pca_2d_matrix", "figure"),
             dash.dependencies.Input("clustering_2d_matrix", "figure"),
             dash.dependencies.Input("ml_axis_picker_x", "value"),
             dash.dependencies.Input("ml_axis_picker_y", "value"),
             dash.dependencies.Input("ml_axis_picker_z", "value")]
        )(self._update_pca_clustering_3d)

        self._app.callback(
            dash.dependencies.Output("heatmap_correlation_selected_cols", "figure"),
            [dash.dependencies.Input("ml_feature_cols_picker", "value")]
        )(self._update_heatmap_correlation_selected_cols)

    # Methods for control panels
    def _update_categories_picker(self, selected_col):

        if not selected_col:
            return [], []

        df = self._data_handler.get_data_frame_original

        all_cat = [{"label": col, "value": col} for col in
                   self._data_handler.get_data_frame_original[selected_col].unique()]
        reserved_cat = DataHandler.get_top_categories(data_frame=df, target_cols=selected_col,
                                                      number_of_reserved=self._default_number_of_reserved)
        return all_cat, reserved_cat

    @staticmethod
    def _update_specific_year_picker(year_range, selected_bar):
        value = selected_bar["points"][0]["x"] if selected_bar and selected_bar["points"] else None
        options = [{"label": year, "value": year} for year in range(year_range[0], year_range[1] + 1)]
        return value, options

    @staticmethod
    def _update_ml_axis_picker(num_of_pc):
        options = [{"label": "P" + str(i + 1), "value": "P" + str(i + 1)} for i in range(num_of_pc)]
        return_settings = []

        for axis_index in range(3):
            return_settings.append(options)
            return_settings.append(options[axis_index]["value"])

        return tuple(return_settings)

    # Methods for graphics
    def _update_bar_year_attack_type_all_fig(self, year_range):
        df = self._get_backend_data_frame_for_viewing(year_range=year_range)
        df = df.groupby(["iyear"]).size().reset_index(name="frequency")

        fig = px.bar(data_frame=df, x="iyear", y="frequency", text="frequency")

        fig.update_traces(opacity=1,
                          textposition="outside",
                          marker=dict(
                              color=df["iyear"],
                              colorscale=DataVisualizer._get_color_scale(len(df["iyear"].unique()),
                                                                         self._default_colors["screen1"]["light"],
                                                                         self._default_colors["screen1"]["dark"])))

        fig.update_layout(autosize=True,
                          showlegend=False,
                          dragmode="select",
                          hovermode="x",
                          margin=go.layout.Margin(l=20, r=20, t=20, b=20),
                          paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color=self._default_colors["screen1"]["light"]),
                          xaxis=dict(
                              title=None,
                              showgrid=False),
                          yaxis=dict(
                              title=None,
                              showticklabels=False,
                              showgrid=False))

        return fig

    def _update_heatmap_weekday_month_year_attack_freq_fig(self, year_range, specified_year):

        # Wait for input fields initialization.
        if not (year_range or specified_year):
            return self._default_plain_fig

        year_range = [specified_year, specified_year] if specified_year else year_range
        df = self._get_backend_data_frame_for_viewing(year_range=year_range)

        month_order = dict(zip(range(len(calendar.month_name)), list(calendar.month_abbr)))
        day_order = dict(zip(range(len(calendar.day_name)), list(calendar.day_abbr)))

        df = df[(df[["iyear", "imonth", "iday"]] != 0).all(axis=1)][["iyear", "imonth", "iday"]]
        df = df.rename(columns={"iyear": "year", "imonth": "month", "iday": "day"})
        df["weekday"] = pd.to_datetime(df[["year", "month", "day"]]).dt.dayofweek
        df = df.groupby(["weekday", "month"]).size()
        df = df.reset_index(name="frequency")
        df = df.pivot(index="weekday", columns="month", values="frequency")
        df = df.rename(index=day_order).rename(columns=month_order)

        annotations = []
        for row_index, row in enumerate(df.values):
            for col_index, cell in enumerate(row):
                annotations.append(dict(
                    showarrow=False,
                    text="<b>" + str(df.values[row_index][col_index]) + "<b>",
                    xref="x",
                    yref="y",
                    x=df.columns[col_index],
                    y=df.index[row_index]))

        fig = go.Figure(
            data=go.Heatmap(
                z=df.values,
                x=df.columns,
                y=df.index,
                colorscale=[[0, self._default_colors["screen1"]["dark"]],
                            [1, self._default_colors["screen1"]["light"]]]))

        fig.update_layout(autosize=True,
                          showlegend=False,
                          margin=go.layout.Margin(l=20, r=20, t=20, b=20),
                          paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color=self._default_colors["screen1"]["light"]),
                          annotations=annotations)

        return fig

    def _update_map_year_attack_type_all_or_specified_fig(self, year_range, specified_year, selected_col, selected_cat):

        # Wait for input fields initialization.
        if not (selected_col and selected_cat and (year_range or specified_year)):
            return self._default_plain_fig

        year_range = [specified_year, specified_year] if specified_year else year_range
        df = self._get_backend_data_frame_for_viewing(year_range=year_range, selected_col=selected_col,
                                                      selected_cat=selected_cat)

        df_m = df[[selected_col, "latitude", "longitude", "nkill", "nwound"]].groupby(
            [selected_col, "latitude", "longitude"]).sum()
        df_m["frequency"] = df.groupby([selected_col, "latitude", "longitude"]).size()
        df = df_m.reset_index()

        fig = px.scatter_mapbox(data_frame=df, lat="latitude", lon="longitude", color=selected_col,
                                size="frequency",
                                zoom=2.2, custom_data=[selected_col, "nkill", "nwound"])

        fig.update_layout(autosize=True,
                          showlegend=False,
                          margin=go.layout.Margin(l=0, r=0, t=0, b=0),
                          mapbox=dict(accesstoken=self._mapbox_token, style=self._mapbox_style))

        return fig

    def _update_pies_kill_wound_nationality_selected_points(self, selected_points, year_range, specified_year,
                                                            selected_col, selected_cat):

        # Wait for input fields initialization.
        if not (selected_col and selected_cat and (year_range or specified_year)):
            return self._default_plain_fig

        df_col_list = [selected_col, "nkill", "nwound", "nkill+wound"]

        if selected_points and selected_points["points"]:
            df = pd.DataFrame([point["customdata"] for point in selected_points["points"]],
                              columns=df_col_list[:-1])
        else:
            year_range = [specified_year, specified_year] if specified_year else year_range
            df = self._get_backend_data_frame_for_viewing(year_range=year_range, selected_col=selected_col,
                                                          selected_cat=selected_cat)

        df["nkill+wound"] = df["nkill"] + df["nwound"]
        df = df[df_col_list].groupby(selected_col).sum().reset_index()

        fig = make_subplots(rows=len(df_col_list[1:]), cols=1,
                            specs=[[{"type": "domain"}] for _ in range(len(df_col_list[1:]))])
        for index, pie_col in enumerate(df_col_list[1:]):
            fig.add_trace(go.Pie(labels=df[selected_col].unique(), values=df[pie_col], name=pie_col), index + 1, 1)

        fig.update_traces(hole=.4,
                          hoverinfo="label+percent+name",
                          marker=dict(
                              colors=DataVisualizer._get_color_scale(len(selected_cat),
                                                                     self._default_colors["screen2"]["light"],
                                                                     self._default_colors["screen2"]["dark"])))

        fig.update_layout(autosize=True,
                          showlegend=False,
                          font=dict(color=self._default_colors["screen2"]["light"]),
                          paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        return fig

    def _update_pca_clustering_matrix(self, trigger, selected_target, selected_cols, num_of_pc, random_state):

        # Wait for input fields initialization.
        if not (selected_target and selected_cols and num_of_pc and random_state):
            return self._default_plain_fig, self._default_plain_fig

        self._pca_clustering_target = selected_target
        all_focused_cols = selected_cols + [selected_target]
        return_list = []

        # Compute PCA
        self._pca_clustering_df["PCA"] = self._data_handler.get_data_frame_original
        self._pca_clustering_df["PCA"] = self._pca_clustering_df["PCA"][all_focused_cols]
        self._pca_clustering_df["PCA"] = self._pca_clustering_df["PCA"].dropna(subset=all_focused_cols).reset_index()

        self._pca_clustering_df["PCA"], total_var = DataHandler.get_pca(data_frame=self._pca_clustering_df["PCA"],
                                                                        target=selected_target,
                                                                        num_of_pc=num_of_pc)

        # Compute Clustering by using self._pca_clustering_df["PCA"]
        self._pca_clustering_df["K-Means Clustering"] = DataHandler.get_clustering(
            data_frame=self._pca_clustering_df["PCA"], target=selected_target,
            random_state=random_state)

        # Create Matrix figures - Meta
        computed_feature_cols = ["P" + str(i + 1) for i in range(num_of_pc)]

        labels = dict(zip(map(str, range(num_of_pc)), computed_feature_cols))
        labels["color"] = selected_target

        # Create Matrix figures
        for key in self._pca_clustering_df.keys():
            return_list.append(
                px.scatter_matrix(
                    self._pca_clustering_df[key],
                    color=self._pca_clustering_df[key][selected_target],
                    dimensions=computed_feature_cols,
                    labels=labels,
                    template="simple_white"
                )
            )

            return_list[-1].update_traces(diagonal_visible=False,
                                          marker_coloraxis=None)

            return_list[-1].update_layout(autosize=True,
                                          showlegend=False,
                                          title=key,
                                          margin=go.layout.Margin(l=0, r=0, t=50, b=0),
                                          font=dict(color=self._default_colors["screen3"]["light"]),
                                          paper_bgcolor="rgba(0,0,0,0)",
                                          plot_bgcolor="rgba(0,0,0,0)")

        return_list.append(html.Span(f"Total Explained Variance: {total_var * 100:.2f}%"))

        return tuple(return_list)

    def _update_pca_clustering_3d(self, pca_matrix_fig, clustering_matrix_fig, axis_x, axis_y, axis_z):

        # Wait for input fields initialization.
        if pca_matrix_fig == self._default_plain_fig or \
                clustering_matrix_fig == self._default_plain_fig or \
                not (axis_x and axis_y and axis_z):
            return self._default_plain_fig, self._default_plain_fig

        return_list = []

        # Create 3D figures
        for key in self._pca_clustering_df.keys():
            return_list.append(
                px.scatter_3d(data_frame=self._pca_clustering_df[key],
                              x=axis_x,
                              y=axis_y,
                              z=axis_z,
                              color=self._pca_clustering_df[key][self._pca_clustering_target],
                              template="simple_white"))

            return_list[-1].update_traces(marker_coloraxis=None)

            return_list[-1].update_layout(autosize=True,
                                          title=key,
                                          showlegend=False,
                                          margin=go.layout.Margin(l=0, r=0, t=50, b=0),
                                          font=dict(color=self._default_colors["screen3"]["light"]),
                                          paper_bgcolor="rgba(0,0,0,0)",
                                          plot_bgcolor="rgba(0,0,0,0)",
                                          scene=dict(
                                              xaxis=dict(showbackground=False),
                                              yaxis=dict(showbackground=False),
                                              zaxis=dict(showbackground=False)))

        return tuple(return_list)

    def _update_heatmap_correlation_selected_cols(self, selected_calc_col):

        # Wait for input fields initialization.
        if not selected_calc_col:
            return self._default_plain_fig

        df = self._data_handler.get_data_frame_original[selected_calc_col].corr()

        fig = go.Figure(data=go.Heatmap(
            z=df.values,
            x=df.columns,
            y=df.index,
            colorscale=[[0, self._default_colors["screen3"]["dark"]],
                        [1, self._default_colors["screen3"]["light"]]]))

        fig.update_layout(autosize=True,
                          showlegend=False,
                          title="Correlation Matrix for Selected Columns",
                          margin=go.layout.Margin(l=0, r=0, t=50, b=0),
                          paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color=self._default_colors["screen3"]["light"]))

        return fig

    def _get_backend_data_frame_for_viewing(self, year_range, selected_col=None, selected_cat=None):
        df = self._data_handler.get_data_frame_original

        if selected_col:
            df = DataHandler.trim_categories(data_frame=df, target_cols=selected_col, designated_list=selected_cat)
        df = df[df["iyear"].between(year_range[0], year_range[1], inclusive=True)]

        return df

    @staticmethod
    def _get_color_scale(steps, c_from, c_to):
        return [color.hex for color in list(Color(c_from).range_to(Color(c_to), steps))]

    def run_server(self):
        self._app.run_server(debug=True)


def main():
    path = "dataset/globalterrorismdb_0718dist.csv"
    visualizer = DataVisualizer.construct_from_csv(path=path)
    visualizer.run_server()


if __name__ == "__main__":
    main()
