import pandas as pd
import dash
import dash_html_components as html
import plotly.express as px
import dash_core_components as dcc
import plotly.graph_objects as go
import calendar

from plotly.subplots import make_subplots
from DataHandler import DataHandler


class DataVisualizer(object):

    def __init__(self, app, data_handler):
        self._app = app
        self._data_handler = data_handler

        self._default_year_range = [1994, 2017]
        self._default_number_of_reserved = 8
        self._default_column_pick = self._data_handler.get_txt_columns[0]
        self._default_pca_pick = "region_txt"

        self._default_legend_style = {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1
        }

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
                    html.H1(children="Project 2: Interactive Visual Analytics Dashboard"),
                    html.Div(children="Yi-Chen Liu © 2020 Copyright held by the owner/author(s)."),

                    html.H5(children="Year Range Picker"),
                    dcc.RangeSlider(
                        id="year_slider",
                        min=self._data_handler.get_data_frame_original["iyear"].min(),
                        max=self._data_handler.get_data_frame_original["iyear"].max(),
                        value=self._default_year_range
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
                        options=[{"label": "View: " + col, "value": col} for col in self._data_handler.get_txt_columns],
                        value=self._default_column_pick,
                        placeholder="Select a column in the dataset"
                    ),

                    html.H5(children="Detail Categories Picker"),
                    dcc.Dropdown(id="categories_picker", multi=True),

                    html.H5(children="PCA Pivot Picker"),
                    dcc.Dropdown(
                        id="pca_pivot_picker",
                        options=[{"label": col, "value": col} for col in
                                 self._data_handler.get_data_frame_original.columns],
                        value=self._default_pca_pick
                    ),

                    html.H5(children="PCA Numeric Columns Picker"),
                    dcc.Dropdown(
                        id="pca_numeric_cols_picker",
                        options=[{"label": col, "value": col} for col in self._data_handler.get_numeric_columns],
                        value=self._data_handler.get_numeric_columns,
                        multi=True
                    ),

                    html.H5(children="K Means Random State Parameter"),
                    dcc.Input(
                        id="ml_random_state_setup",
                        type="number",
                        value=5
                    ),

                    html.H5(children="K Means 3D Scatter Axis Picker"),
                    dcc.Checklist(id="ml_axis_picker")
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
                                    dcc.Graph(id="bar_year_attack_type_all_fig", className="graph_style")
                                ]
                            ),

                            html.Div(
                                id="screen01",
                                children=[
                                    dcc.Graph(id="heatmap_weekday_month_year_attack_freq_fig", className="graph_style")
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
                                    dcc.Graph(id="map_year_attack_type_all_or_specified_fig", className="graph_style")
                                ]
                            ),

                            html.Div(
                                id="screen11",
                                children=[
                                    dcc.Graph(id="pies_kill_wound_nationality_selected_points", className="graph_style")
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
                                    dcc.Graph(id="clustering", className="graph_style")
                                ]
                            ),
                            html.Div(
                                id="screen21",
                                children=[
                                    html.Div(
                                        id="screen210",
                                        children=[
                                            dcc.Graph(id="pca", className="graph_style"),
                                        ]
                                    ),
                                    html.Div(
                                        id="screen211",
                                        children=[
                                            dcc.Graph(id="heatmap_correlation_selected_cols", className="graph_style")
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

            all_cat = [{"label": col, "value": col} for col in self._data_handler.get_data_frame_original[selected_col].unique()]
            reserved_cat = DataHandler.get_top_categories(data_frame=df, target_cols=selected_col, number_of_reserved=self._default_number_of_reserved)
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
            [dash.dependencies.Input("pca_numeric_cols_picker", "value")]
        )
        def update_ml_axis_picker(selected_features):
            return [{"label": col, "value": col} for col in selected_features], selected_features[:3]

        @self._app.callback(
            dash.dependencies.Output("bar_year_attack_type_all_fig", "figure"),
            [dash.dependencies.Input("year_slider", "value"),
             dash.dependencies.Input("column_picker", "value"),
             dash.dependencies.Input("categories_picker", "value")])
        def update_bar_year_attack_type_all_fig(year_range, selected_col, selected_cat):
            df = self.get_backend_data_frame_for_viewing(year_range=year_range, selected_col=selected_col, selected_cat=selected_cat)
            df = df.groupby(["iyear", selected_col]).size().reset_index(name="frequency")

            fig = px.bar(data_frame=df, x="iyear", y="frequency", color=selected_col, custom_data=["iyear", selected_col])
            fig.update_layout(autosize=True, title="Yearly Accumulated Attacks and Types")

            return fig

        @self._app.callback(
            dash.dependencies.Output("heatmap_weekday_month_year_attack_freq_fig", "figure"),
            [dash.dependencies.Input("year_slider", "value"),
             dash.dependencies.Input("specific_year_picker", "value"),
             dash.dependencies.Input("column_picker", "value"),
             dash.dependencies.Input("categories_picker", "value")])
        def update_heatmap_weekday_month_year_attack_freq_fig(year_range, specified_year, selected_col, selected_cat):
            year_range = [specified_year, specified_year] if specified_year else year_range
            df = self.get_backend_data_frame_for_viewing(year_range=year_range, selected_col=selected_col,
                                                         selected_cat=selected_cat)

            month_order = dict(zip(range(len(calendar.month_name)), list(calendar.month_abbr)))
            day_order = dict(zip(range(len(calendar.day_name)), list(calendar.day_abbr)))

            df = df[(df[["iyear", "imonth", "iday"]] != 0).all(axis=1)][["iyear", "imonth", "iday"]]
            df = df.rename(columns={"iyear": "year", "imonth": "month", "iday": "day"})
            df["weekday"] = pd.to_datetime(df[["year", "month", "day"]]).dt.dayofweek
            df = df.groupby(["weekday", "month"]).size().reset_index(name="frequency")
            df = df.pivot(index="weekday", columns="month", values="frequency")
            df = df.rename(index=day_order).rename(columns=month_order)

            fig = px.imshow(df.values, labels=dict(x="Month of Year", y="Day of Week", color="Attack Times"), x=df.columns, y=df.index)
            fig.update_layout(autosize=True, title="Yearly Accumulated Attacks and Types")

            return fig

        @self._app.callback(
            dash.dependencies.Output("map_year_attack_type_all_or_specified_fig", "figure"),
            [dash.dependencies.Input("year_slider", "value"),
             dash.dependencies.Input("specific_year_picker", "value"),
             dash.dependencies.Input("column_picker", "value"),
             dash.dependencies.Input("categories_picker", "value")])
        def update_map_year_attack_type_all_or_specified_fig(year_range, specified_year, selected_col, selected_cat):
            year_range = [specified_year, specified_year] if specified_year else year_range
            df = self.get_backend_data_frame_for_viewing(year_range=year_range, selected_col=selected_col, selected_cat=selected_cat)

            df_m = df[[selected_col, "latitude", "longitude", "nkill", "nwound"]].groupby([selected_col, "latitude", "longitude"]).sum()
            df_m["frequency"] = df.groupby([selected_col, "latitude", "longitude"]).size()
            df = df_m.reset_index()

            fig = px.scatter_mapbox(data_frame=df, lat="latitude", lon="longitude", color=selected_col, size="frequency",
                                    zoom=2, custom_data=[selected_col, "nkill", "nwound"], mapbox_style="open-street-map")
            fig.update_layout(autosize=True, title="Attacks Map by Viewing " + selected_col)

            return fig

        @self._app.callback(
            dash.dependencies.Output("pies_kill_wound_nationality_selected_points", "figure"),
            [dash.dependencies.Input("map_year_attack_type_all_or_specified_fig", "selectedData"),
             dash.dependencies.Input("year_slider", "value"),
             dash.dependencies.Input("specific_year_picker", "value"),
             dash.dependencies.Input("column_picker", "value"),
             dash.dependencies.Input("categories_picker", "value")])
        def update_pies_kill_wound_nationality_selected_points(selected_points, year_range, specified_year, selected_col, selected_cat):
            df_col_list = [selected_col, "nkill", "nwound", "nkill+wound"]

            if selected_points and selected_points["points"]:
                df = pd.DataFrame([point["customdata"] for point in selected_points["points"]], columns=df_col_list[:-1])
            else:
                year_range = [specified_year, specified_year] if specified_year else year_range
                df = self.get_backend_data_frame_for_viewing(year_range=year_range, selected_col=selected_col,
                                                             selected_cat=selected_cat)

            df["nkill+wound"] = df["nkill"] + df["nwound"]
            df = df[df_col_list].groupby(selected_col).sum().reset_index()

            fig = make_subplots(rows=len(df_col_list[1:]), cols=1, specs=[[{"type": "domain"}] for _ in range(len(df_col_list[1:]))])
            for index, pies_col in enumerate(df_col_list[1:]):
                fig.add_trace(go.Pie(labels=df[selected_col].unique(), values=df[pies_col], name=pies_col), index + 1, 1)

            fig.update_layout(autosize=True, title="Detail View for Hovered Instance")
            fig.update_traces(hole=.4, hoverinfo="label+percent+name")

            return fig

        @self._app.callback(
            dash.dependencies.Output("clustering", "figure"),
            [dash.dependencies.Input("pca_pivot_picker", "value"),
             dash.dependencies.Input("pca_numeric_cols_picker", "value"),
             dash.dependencies.Input("ml_random_state_setup", "value"),
             dash.dependencies.Input("ml_axis_picker", "value")])
        def update_clustering(selected_label_col, selected_calc_col, random, axis):
            selected_label_col = selected_label_col if selected_label_col else self._default_pca_pick
            self._default_pca_pick = selected_label_col

            df_clustering = self._data_handler.get_data_frame_clustering(selected_label_col, selected_calc_col, random)
            df_clustering = DataHandler.trim_categories(data_frame=df_clustering, target_cols=selected_label_col)

            fig = px.scatter_3d(data_frame=df_clustering, x=axis[0], y=axis[1], z=axis[2], color=selected_label_col)
            fig.update_layout(legend=self._default_legend_style, autosize=True, title="K Means Clustering")
            return fig

        @self._app.callback(
            dash.dependencies.Output("pca", "figure"),
            dash.dependencies.Output("heatmap_correlation_selected_cols", "figure"),
            [dash.dependencies.Input("pca_pivot_picker", "value"),
             dash.dependencies.Input("pca_numeric_cols_picker", "value")])
        def update_pca_and_heatmap_correlation_selected_cols(selected_label_col, selected_calc_col):
            selected_label_col = selected_label_col if selected_label_col else self._default_pca_pick
            self._default_pca_pick = selected_label_col

            df_pca = self._data_handler.get_data_frame_pca(selected_label_col, selected_calc_col)
            df_pca = DataHandler.trim_categories(data_frame=df_pca, target_cols=selected_label_col)

            fig_pca = px.scatter_3d(data_frame=df_pca, x="P1", y="P2", z="P3", color=selected_label_col)
            fig_pca.update_layout(legend=self._default_legend_style, autosize=True, title="PCA")

            df_corr = self._data_handler.get_data_frame_original[selected_calc_col].corr()
            fig_corr = px.imshow(df_corr.values, labels=dict(color="Corr"), x=df_corr.columns, y=df_corr.index)

            return fig_pca, fig_corr

    def get_backend_data_frame_for_viewing(self, year_range, selected_col, selected_cat):
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
