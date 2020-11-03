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

    def __init__(self, app, data_handler):
        self._app = app
        self._data_handler = data_handler

        self._default_year_range = [1994, 2017]
        self._default_number_of_reserved = 8
        self._default_column_pick = self._data_handler.get_txt_columns[0]
        self._default_pca_pick = "region_txt"

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

        self._default_color_set = {
            "screen1": {"light": "#C4DBFF", "dark": "#202E45"},
            "screen2": {"light": "#C1FFFA", "dark": "#10523E"},
            "screen3": {"light": "#FDDFD5", "dark": "#3A0000"}
        }

        self._default_color_scale = lambda steps, c_from, c_to: [color.hex for color in list(Color(c_from).range_to(Color(c_to), steps))]

        self._mapbox_access_token = "pk.eyJ1IjoibGVvMTk5MDA3MjMiLCJhIjoiY2toMTM0NGVqMGFzdzJycnh0M3RpNnd6cSJ9.HI8SD_-Mbl2Cwa2c-W9PNA"
        self._mapbox_style = "mapbox://styles/leo19900723/ckh13dngl0b6s19nw1vm9h9bq"

        self.set_layout()

    @classmethod
    def construct_from_csv(cls, path):
        external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

        return DataVisualizer(
            app=dash.Dash(__name__, external_stylesheets=external_stylesheets),
            data_handler=DataHandler.construct_from_csv(path)
        )

    def set_layout(self):

        self._app.layout = html.Div(id="main", children=[
            html.Div(
                id="side_bar",
                children=[
                    html.Div(
                        id="side_bar_top",
                        children=[
                            html.H1(children="Project 2: Interactive Visual Analytics Dashboard"),
                            html.Div(children="Yi-Chen Liu © 2020 Copyright held by the owner/author(s).")
                        ]
                    ),

                    html.Div(
                        id="side_bar_bottom",
                        children=[
                            html.Div(
                                id="side_bar_bottom0",
                                children=[
                                    html.H5(children="Year Range Picker"),
                                    html.Div(children=[
                                            dcc.RangeSlider(
                                                id="year_slider",
                                                min=self._data_handler.get_data_frame_original["iyear"].min(),
                                                max=self._data_handler.get_data_frame_original["iyear"].max(),
                                                value=self._default_year_range
                                            )
                                        ],
                                        style={"padding-left": "5%", "padding-right": "5%"}
                                    ),

                                    html.H5(children="Specific Year Picker"),
                                    dcc.Dropdown(
                                        id="specific_year_picker",
                                        options=[{"label": year, "value": year} for year in
                                                 self._data_handler.get_data_frame_original["iyear"].unique()],
                                        placeholder="Select a year"
                                    ),

                                    html.H5(children="Column Picker"),
                                    dcc.Dropdown(
                                        id="column_picker",
                                        options=[{"label": "View: " + col, "value": col} for col in
                                                 self._data_handler.get_txt_columns],
                                        value=self._default_column_pick,
                                        placeholder="Select a column in the dataset"
                                    ),

                                    html.H5(children="Detail Categories Picker"),
                                    dcc.Dropdown(id="categories_picker", multi=True)
                                ]
                            ),

                            html.Div(
                                id="side_bar_bottom1",
                                children=[
                                    html.H5(children="PCA/ K-Means Target Picker"),
                                    dcc.Dropdown(
                                        id="ml_target_picker",
                                        options=[{"label": col, "value": col} for col in
                                                 self._data_handler.get_data_frame_original.columns],
                                        value=self._default_pca_pick
                                    ),

                                    html.H5(children="PCA/ K-Means Feature Columns Picker"),
                                    dcc.Dropdown(
                                        id="ml_feature_cols_picker",
                                        options=[{"label": col, "value": col} for col in
                                                 self._data_handler.get_numeric_columns],
                                        value=self._data_handler.get_numeric_columns,
                                        multi=True
                                    ),

                                    html.H5(children="K-Means Random State Parameter"),
                                    dcc.Input(
                                        id="ml_random_state_setup",
                                        type="number",
                                        value=5
                                    ),

                                    html.H5(children="K-Means 3D Scatter Axis Picker"),
                                    dcc.Checklist(id="ml_axis_picker")
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
                                    dcc.Graph(id="clustering",
                                              figure=self._default_plain_fig,
                                              className="graph_style")
                                ]
                            ),
                            html.Div(
                                id="screen21",
                                children=[
                                    html.Div(
                                        id="screen210",
                                        children=[
                                            dcc.Graph(id="pca",
                                                      figure=self._default_plain_fig,
                                                      className="graph_style"),
                                        ]
                                    ),
                                    html.Div(
                                        id="screen211",
                                        children=[
                                            dcc.Graph(id="heatmap_correlation_selected_cols",
                                                      figure=self._default_plain_fig,
                                                      className="graph_style")
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            )
        ])

        @self._app.callback(
            dash.dependencies.Output("categories_picker", "options"),
            dash.dependencies.Output("categories_picker", "value"),
            [dash.dependencies.Input("column_picker", "value")]
        )
        def update_categories_picker(selected_col):
            df = self._data_handler.get_data_frame_original

            all_cat = [{"label": col, "value": col} for col in
                       self._data_handler.get_data_frame_original[selected_col].unique()]
            reserved_cat = DataHandler.get_top_categories(data_frame=df, target_cols=selected_col,
                                                          number_of_reserved=self._default_number_of_reserved)
            return all_cat, reserved_cat

        @self._app.callback(
            dash.dependencies.Output("specific_year_picker", "value"),
            [dash.dependencies.Input("bar_year_attack_type_all_fig", "selectedData")]
        )
        def update_specific_year_picker(selected_bar):
            return selected_bar["points"][0]["customdata"][0] if selected_bar and selected_bar["points"] else None

        @self._app.callback(
            dash.dependencies.Output("ml_axis_picker", "options"),
            dash.dependencies.Output("ml_axis_picker", "value"),
            [dash.dependencies.Input("ml_feature_cols_picker", "value")]
        )
        def update_ml_axis_picker(selected_features):
            return [{"label": col, "value": col} for col in selected_features], selected_features[:3]

        @self._app.callback(
            dash.dependencies.Output("bar_year_attack_type_all_fig", "figure"),
            [dash.dependencies.Input("year_slider", "value"),
             dash.dependencies.Input("column_picker", "value")])
        def update_bar_year_attack_type_all_fig(year_range, selected_col):
            df = self.get_backend_data_frame_for_viewing(year_range=year_range, selected_col=selected_col)
            df = df.groupby(["iyear"]).size().reset_index(name="frequency")

            fig = px.bar(data_frame=df, x="iyear", y="frequency", text="frequency", custom_data=["iyear"])

            fig.update_traces(opacity=1,
                              textposition="outside",
                              marker=dict(
                                  color=df["iyear"],
                                  colorscale=self._default_color_scale(len(df["iyear"].unique()), self._default_color_set["screen1"]["light"], self._default_color_set["screen1"]["dark"]))
                              )

            fig.update_layout(autosize=True,
                              showlegend=False,
                              dragmode="select",
                              margin=go.layout.Margin(l=20, r=20, t=20, b=20),
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color=self._default_color_set["screen1"]["light"]),
                              xaxis=dict(
                                  title=None,
                                  showgrid=False),
                              yaxis=dict(
                                  title=None,
                                  showticklabels=False,
                                  showgrid=False),
                              )

            return fig

        @self._app.callback(
            dash.dependencies.Output("heatmap_weekday_month_year_attack_freq_fig", "figure"),
            [dash.dependencies.Input("year_slider", "value"),
             dash.dependencies.Input("specific_year_picker", "value"),
             dash.dependencies.Input("column_picker", "value")])
        def update_heatmap_weekday_month_year_attack_freq_fig(year_range, specified_year, selected_col):

            # Wait for input fields initialization.
            if not (selected_col and (year_range or specified_year)):
                return self._default_plain_fig

            year_range = [specified_year, specified_year] if specified_year else year_range
            df = self.get_backend_data_frame_for_viewing(year_range=year_range, selected_col=selected_col)

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
            for n, row in enumerate(df.values):
                for m, val in enumerate(row):
                    annotations.append(dict(
                            showarrow=False,
                            text="<b>" + str(df.values[n][m]) + "<b>",
                            xref="x",
                            yref="y",
                            x=df.columns[m],
                            y=df.index[n],
                        ))

            fig = go.Figure(data=go.Heatmap(
                z=df.values,
                x=df.columns,
                y=df.index,
                colorscale=[[0, self._default_color_set["screen1"]["dark"]], [1, self._default_color_set["screen1"]["light"]]])
            )

            fig.update_layout(autosize=True,
                              showlegend=False,
                              margin=go.layout.Margin(l=20, r=20, t=20, b=20),
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color=self._default_color_set["screen1"]["light"]),
                              annotations=annotations
                              )

            return fig

        @self._app.callback(
            dash.dependencies.Output("map_year_attack_type_all_or_specified_fig", "figure"),
            [dash.dependencies.Input("year_slider", "value"),
             dash.dependencies.Input("specific_year_picker", "value"),
             dash.dependencies.Input("column_picker", "value"),
             dash.dependencies.Input("categories_picker", "value")])
        def update_map_year_attack_type_all_or_specified_fig(year_range, specified_year, selected_col, selected_cat):

            # Wait for input fields initialization.
            if not (selected_col and selected_cat and (year_range or specified_year)):
                return self._default_plain_fig

            year_range = [specified_year, specified_year] if specified_year else year_range
            df = self.get_backend_data_frame_for_viewing(year_range=year_range, selected_col=selected_col,
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
                              mapbox=dict(accesstoken=self._mapbox_access_token, style=self._mapbox_style)
                              )

            return fig

        @self._app.callback(
            dash.dependencies.Output("pies_kill_wound_nationality_selected_points", "figure"),
            [dash.dependencies.Input("map_year_attack_type_all_or_specified_fig", "selectedData"),
             dash.dependencies.Input("year_slider", "value"),
             dash.dependencies.Input("specific_year_picker", "value"),
             dash.dependencies.Input("column_picker", "value"),
             dash.dependencies.Input("categories_picker", "value")])
        def update_pies_kill_wound_nationality_selected_points(selected_points, year_range, specified_year,
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
                df = self.get_backend_data_frame_for_viewing(year_range=year_range, selected_col=selected_col,
                                                             selected_cat=selected_cat)

            df["nkill+wound"] = df["nkill"] + df["nwound"]
            df = df[df_col_list].groupby(selected_col).sum().reset_index()

            fig = make_subplots(rows=len(df_col_list[1:]), cols=1,
                                specs=[[{"type": "domain"}] for _ in range(len(df_col_list[1:]))])
            for index, pies_col in enumerate(df_col_list[1:]):
                fig.add_trace(go.Pie(labels=df[selected_col].unique(), values=df[pies_col], name=pies_col), index + 1,
                              1)

            fig.update_traces(hole=.4,
                              hoverinfo="label+percent+name",
                              marker=dict(colors=self._default_color_scale(len(selected_cat), self._default_color_set["screen2"]["light"], self._default_color_set["screen2"]["dark"])))

            fig.update_layout(autosize=True,
                              showlegend=False,
                              font=dict(color=self._default_color_set["screen2"]["light"]),
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)"
                              )
            return fig

        @self._app.callback(
            dash.dependencies.Output("clustering", "figure"),
            [dash.dependencies.Input("ml_target_picker", "value"),
             dash.dependencies.Input("ml_feature_cols_picker", "value"),
             dash.dependencies.Input("ml_random_state_setup", "value"),
             dash.dependencies.Input("ml_axis_picker", "value")])
        def update_clustering(selected_label_col, selected_calc_col, random, axis):

            # Wait for input fields initialization.
            if not(selected_label_col and selected_calc_col and random and axis):
                return self._default_plain_fig

            selected_label_col = selected_label_col if selected_label_col else self._default_pca_pick
            self._default_pca_pick = selected_label_col

            df_clustering = self._data_handler.get_data_frame_clustering(selected_label_col, selected_calc_col, random)
            df_clustering = DataHandler.trim_categories(data_frame=df_clustering, target_cols=selected_label_col)

            fig = px.scatter_3d(data_frame=df_clustering, x=axis[0], y=axis[1], z=axis[2], color=selected_label_col, symbol=selected_label_col)

            axis_template = {
                "showbackground": False,
                "gridcolor": self._default_color_set["screen3"]["light"],
                "zerolinecolor": self._default_color_set["screen3"]["light"],
            }

            fig.update_layout(autosize=True,
                              title="K-Means Clustering",
                              showlegend=False,
                              font=dict(color=self._default_color_set["screen3"]["light"]),
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              scene=dict(
                                  xaxis=axis_template,
                                  yaxis=axis_template,
                                  zaxis=axis_template)
                              )

            return fig

        @self._app.callback(
            dash.dependencies.Output("pca", "figure"),
            [dash.dependencies.Input("ml_target_picker", "value"),
             dash.dependencies.Input("ml_feature_cols_picker", "value")])
        def update_pca(selected_label_col, selected_calc_col):

            # Wait for input fields initialization.
            if not(selected_label_col and selected_calc_col):
                return self._default_plain_fig

            selected_label_col = selected_label_col if selected_label_col else self._default_pca_pick
            self._default_pca_pick = selected_label_col

            df_pca = self._data_handler.get_data_frame_pca(selected_label_col, selected_calc_col)
            df_pca = DataHandler.trim_categories(data_frame=df_pca, target_cols=selected_label_col)

            fig = px.scatter_3d(data_frame=df_pca, x="P1", y="P2", z="P3", color=selected_label_col, symbol=selected_label_col)

            axis_template = {
                "showbackground": False,
                "gridcolor": self._default_color_set["screen3"]["light"],
                "zerolinecolor": self._default_color_set["screen3"]["light"],
            }

            fig.update_layout(autosize=True,
                              title="PCA",
                              showlegend=False,
                              font=dict(color=self._default_color_set["screen3"]["light"]),
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              scene=dict(
                                  xaxis=axis_template,
                                  yaxis=axis_template,
                                  zaxis=axis_template)
                              )

            return fig

        @self._app.callback(
            dash.dependencies.Output("heatmap_correlation_selected_cols", "figure"),
            [dash.dependencies.Input("ml_feature_cols_picker", "value")])
        def update_heatmap_correlation_selected_cols(selected_calc_col):

            # Wait for input fields initialization.
            if not selected_calc_col:
                return self._default_plain_fig

            df = self._data_handler.get_data_frame_original[selected_calc_col].corr()
            fig = px.imshow(df.values, labels=dict(color="Corr"), x=df.columns, y=df.index)

            fig = go.Figure(data=go.Heatmap(
                z=df.values,
                x=df.columns,
                y=df.index,
                colorscale=[[0, self._default_color_set["screen3"]["dark"]], [1, self._default_color_set["screen3"]["light"]]])
            )

            fig.update_layout(autosize=True,
                              showlegend=False,
                              margin=go.layout.Margin(l=20, r=20, t=20, b=20),
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color=self._default_color_set["screen3"]["light"])
                              )

            return fig

    def get_backend_data_frame_for_viewing(self, year_range, selected_col, selected_cat=None):
        df = self._data_handler.get_data_frame_original

        df = DataHandler.trim_categories(data_frame=df, target_cols=selected_col, designated_list=selected_cat)
        df = df[df["iyear"].between(year_range[0], year_range[1], inclusive=True)]

        return df

    def run_server(self):
        self._app.run_server(debug=True)


def main():
    path = "dataset/globalterrorismdb_0718dist.csv"
    visualizer = DataVisualizer.construct_from_csv(path=path)
    visualizer.run_server()


if __name__ == "__main__":
    main()
