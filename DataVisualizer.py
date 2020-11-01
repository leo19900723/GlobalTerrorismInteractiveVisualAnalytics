import pandas as pd
import dash
import dash_html_components as html
import plotly.express as px
import dash_core_components as dcc

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

                    html.H5(children="PCA Category View Picker"),
                    dcc.Dropdown(
                        id="pca_view_picker",
                        options=[{"label": col, "value": col} for col in
                                 self._data_handler.get_data_frame_original.columns],
                        value=self._default_pca_pick
                    )
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
                                children="Heat map YO!!!!"
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
                                    html.Div(
                                        id="screen110",
                                        children=[
                                            dcc.Graph(id="scatter_kill_wound_selected_group_and_year", className="graph_style")
                                        ]
                                    ),
                                    html.Div(
                                        id="screen111",
                                        children=[
                                            dcc.Graph(id="pie_year_target_selected_group_fig", className="graph_style")
                                        ]
                                    )
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
                                    dcc.Graph(id="pca", className="graph_style")
                                ]
                            ),
                            html.Div(
                                id="screen21",
                                children="ML Prediction YO!!"
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
            reserved_cat = DataHandler.get_top_categories(data_frame=df, target_col=selected_col, number_of_reserved=self._default_number_of_reserved)
            return all_cat, reserved_cat

        @self._app.callback(
            dash.dependencies.Output("specific_year_picker", "value"),
            [dash.dependencies.Input("bar_year_attack_type_all_fig", "selectedData")]
        )
        def update_specific_year_picker(selected_bar):
            return selected_bar["points"][0]["customdata"][0] if selected_bar and selected_bar["points"] else None

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
            dash.dependencies.Output("map_year_attack_type_all_or_specified_fig", "figure"),
            [dash.dependencies.Input("year_slider", "value"),
             dash.dependencies.Input("specific_year_picker", "value"),
             dash.dependencies.Input("column_picker", "value"),
             dash.dependencies.Input("categories_picker", "value")])
        def update_map_year_attack_type_all_or_specified_fig(year_range, specified_year, selected_col, selected_cat):
            year_range = [specified_year, specified_year] if specified_year else year_range

            df = self.get_backend_data_frame_for_viewing(year_range=year_range, selected_col=selected_col, selected_cat=selected_cat)
            df = df.groupby([selected_col, "latitude", "longitude"]).size().reset_index(name="frequency")

            fig = px.scatter_mapbox(data_frame=df, lat="latitude", lon="longitude", color=selected_col, size="frequency",
                                    zoom=2, mapbox_style="open-street-map")
            fig.update_layout(autosize=True, title="Attacks Map by Viewing " + selected_col)

            return fig

        @self._app.callback(
            dash.dependencies.Output("scatter_kill_wound_selected_group_and_year", "figure"),
            [dash.dependencies.Input("map_year_attack_type_all_or_specified_fig", "selectedData"),
             dash.dependencies.Input("year_slider", "value"),
             dash.dependencies.Input("specific_year_picker", "value"),
             dash.dependencies.Input("column_picker", "value"),
             dash.dependencies.Input("categories_picker", "value")])
        def update_scatter_kill_wound_selected_group_and_year(selected_points, year_range, specified_year, selected_col, selected_cat):
            year_range = [specified_year, specified_year] if specified_year else year_range
            df = self.get_backend_data_frame_for_viewing(year_range=year_range, selected_col=selected_col,
                                                         selected_cat=selected_cat)

            if selected_points and selected_points["points"]:
                key_list = ["latitude", "longitude"]
                df = pd.concat([df[df[key_list].isin([point["lat"], point["lon"]]).all(axis=1)] for point in selected_points["points"]]).reset_index()

            df = df.groupby([selected_col, "nkill", "nwound"]).size().reset_index(name="frequency")
            fig = px.scatter(data_frame=df, x="nkill", y="nwound", color=selected_col)
            fig.update_layout(autosize=True, title="Detail View for Hovered Instance")

            return fig

        @self._app.callback(
            dash.dependencies.Output("pie_year_target_selected_group_fig", "figure"),
            [dash.dependencies.Input("year_slider", "value"),
             dash.dependencies.Input("map_year_attack_type_all_or_specified_fig", "selectedData"),
             dash.dependencies.Input("categories_picker", "value")])
        def update_pie_attack_target_selected_group(year_range, selected_points, selected_group):
            if selected_points and selected_points["points"]:
                df_selected_points = pd.DataFrame([row["customdata"] for row in selected_points["points"]],
                                                  columns=["gname", "latitude", "longitude"])
            else:
                df_selected_points = self._data_handler.get_data_frame_original
                df_selected_points = df_selected_points[df_selected_points["gname"].isin(selected_group)][
                    ["gname", "latitude", "longitude"]]

            df = self._data_handler.get_data_frame_original
            df = df[df["iyear"].between(year_range[0], year_range[1], inclusive=True)]
            df = df[df["gname"].isin(selected_group)]
            df = df[df["latitude"].isin(df_selected_points["latitude"])]
            df = df[df["longitude"].isin(df_selected_points["longitude"])]

            df_target = df.groupby(["targtype1_txt"]).size().reset_index(name="frequency")

            fig_pie = px.pie(data_frame=df_target, values="frequency", names="targtype1_txt", template="ggplot2")
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(showlegend=False, autosize=True,
                                  title="Target Type by Selected Attacks")

            return fig_pie

        @self._app.callback(
            dash.dependencies.Output("pca", "figure"),
            [dash.dependencies.Input("pca_view_picker", "value")])
        def update_pca(picked_col):
            picked_col = picked_col if picked_col else self._default_pca_pick
            self._default_pca_pick = picked_col

            df_pca = self._data_handler.get_data_frame_pca(picked_col)
            df_pca = DataHandler.trim_categories(data_frame=df_pca, target_col=picked_col)

            fig = px.scatter(data_frame=df_pca, x="x", y="y", color=picked_col)
            fig.update_layout(legend=self._default_legend_style, autosize=True,
                              title="PCA")
            return fig

    def get_backend_data_frame_for_viewing(self, year_range, selected_col, selected_cat):
        df = self._data_handler.get_data_frame_original

        df = DataHandler.trim_categories(data_frame=df, target_col=selected_col, designated_list=selected_cat)
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
