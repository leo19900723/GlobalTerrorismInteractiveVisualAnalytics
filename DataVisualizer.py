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

        self._default_year_range = [2000, 2017]
        self._default_group_list = self._data_handler.get_dataframe_original["gname"].value_counts().head(8).index
        self._default_figure_template = "ggplot2"
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
            html.H1(children="Project 2: Interactive Visual Analytics Dashboard"),
            html.Div(children="Yi-Chen Liu © 2020 Copyright held by the owner/author(s)."),

            html.Div(id="row0", children=[
                html.Div(
                    children=[
                        dcc.RangeSlider(
                            id="year_slider",
                            min=self._data_handler.get_dataframe_original["iyear"].min(),
                            max=self._data_handler.get_dataframe_original["iyear"].max(),
                            marks={str(year): str(year) for year in
                                   self._data_handler.get_dataframe_original["iyear"].unique()},
                            value=self._default_year_range
                        ),
                        html.Br(),
                    ],
                    className="block_style"
                ),
            ]),

            html.Div(id="row1", children=[
                html.Div(
                    id="row10",
                    children=[
                        html.H5(children="Control Panel"),
                        dcc.Dropdown(
                            id="group_selector",
                            options=[{"label": group, "value": group} for group in
                                     self._data_handler.get_dataframe_original["gname"].unique()],
                            value=self._default_group_list,
                            multi=True
                        )
                    ],
                    className="block_style"
                ),
                html.Div(
                    id="row11",
                    children=[
                        html.H5(children="Attacks Map by Groups"),
                        dcc.Graph(id="map_year_attack_group_fig")
                    ],
                    className="block_style"
                ),
            ]),

            html.Div(
                id="row2",
                children=[
                    html.Div(
                        id="row20",
                        children=[
                            html.H5(children="Yearly Attacks by Groups"),
                            dcc.Graph(id="line_year_attack_group_fig")
                        ],
                        className="block_style"
                    ),
                    html.Div(
                        id="row21",
                        children=[
                            html.H5(children="Target Type by Selected Attacks"),
                            dcc.Graph(id="pie_year_target_selected_group_fig")
                        ],
                        className="block_style"
                    )
                ])
        ])

        @self._app.callback(
            dash.dependencies.Output("map_year_attack_group_fig", "figure"),
            [dash.dependencies.Input("year_slider", "value"),
             dash.dependencies.Input("group_selector", "value")])
        def update_map_year_attack_group_fig(year_range, selected_group):
            df = self._data_handler.get_dataframe_original
            df = df[df["iyear"].between(year_range[0], year_range[1], inclusive=True)]
            df = df[df["gname"].isin(selected_group)]
            df = df.groupby(["gname", "latitude", "longitude"]).size().reset_index(name="frequency")

            fig = px.scatter_mapbox(data_frame=df, lat="latitude", lon="longitude", color="gname", size="frequency",
                                    zoom=2, custom_data=["gname", "latitude", "longitude"],
                                    color_continuous_scale=px.colors.cyclical.IceFire, mapbox_style="open-street-map",
                                    template=self._default_figure_template)
            fig.update_layout(legend=self._default_legend_style)
            return fig

        @self._app.callback(
            dash.dependencies.Output("line_year_attack_group_fig", "figure"),
            [dash.dependencies.Input("year_slider", "value"), dash.dependencies.Input("group_selector", "value")])
        def update_line_year_attack_group_fig(year_range, selected_group):
            df = self._data_handler.get_dataframe_original
            df = df[df["iyear"].between(year_range[0], year_range[1], inclusive=True)]
            df = df[df["gname"].isin(selected_group)]
            df = df.groupby(["iyear", "gname"]).size().reset_index(name="frequency")

            fig = px.line(data_frame=df, x="iyear", y="frequency", color="gname",
                          template=self._default_figure_template)
            fig.update_layout(legend=self._default_legend_style)
            return fig

        @self._app.callback(
            dash.dependencies.Output("pie_year_target_selected_group_fig", "figure"),
            [dash.dependencies.Input("year_slider", "value"),
             dash.dependencies.Input("map_year_attack_group_fig", "selectedData"),
             dash.dependencies.Input("group_selector", "value")])
        def update_pie_year_weapon_target_selected_group_fig(year_range, selected_points, selected_group):
            if selected_points:
                df_selected_points = pd.DataFrame([row["customdata"] for row in selected_points["points"]],
                                                  columns=["gname", "latitude", "longitude"])
            else:
                df_selected_points = self._data_handler.get_dataframe_original
                df_selected_points = df_selected_points[df_selected_points["gname"].isin(selected_group)][
                    ["gname", "latitude", "longitude"]]

            df = self._data_handler.get_dataframe_original
            df = df[df["iyear"].between(year_range[0], year_range[1], inclusive=True)]
            df = df[df["gname"].isin(df_selected_points["gname"])]
            df = df[df["latitude"].isin(df_selected_points["latitude"])]
            df = df[df["longitude"].isin(df_selected_points["longitude"])]

            df_target = df.groupby(["targtype1_txt"]).size().reset_index(name="frequency")

            fig = px.pie(data_frame=df_target, values=df_target["frequency"], template=self._default_figure_template)
            fig.update_layout(showlegend=False)
            return fig

    def run_server(self):
        self._app.run_server(debug=True)


def main():
    path = "dataset/globalterrorismdb_0718dist.csv"
    visualizer = DataVisualizer.construct_from_csv(path=path)
    visualizer.run_server()


if __name__ == "__main__":
    main()
