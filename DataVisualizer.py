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
        self._default_group_list = self._data_handler.get_data_frame_original["gname"].value_counts().head(8).index
        self._default_bar_pick = self._data_handler.get_txt_categories[0]
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
        self._app.layout = html.Div(id="main", className="block_style", children=[
            html.H1(children="Project 2: Interactive Visual Analytics Dashboard"),
            html.Div(children="Yi-Chen Liu © 2020 Copyright held by the owner/author(s)."),

            html.Div(id="row0", children=[
                html.Div(
                    id="row00",
                    children=[
                        dcc.RangeSlider(
                            id="year_slider",
                            min=self._data_handler.get_data_frame_original["iyear"].min(),
                            max=self._data_handler.get_data_frame_original["iyear"].max(),
                            marks={str(year): str(year) for year in
                                   self._data_handler.get_data_frame_original["iyear"].unique()},
                            value=self._default_year_range
                        ),
                    ]
                ),
            ]),

            html.Div(id="row1", children=[
                html.Div(
                    id="row10",
                    children=[
                        dcc.Loading(children=[dcc.Graph(id="bar_year_attack_type_all_fig", className="graph_style")])
                    ]
                ),
                html.Div(
                    id="row11",
                    children=[
                        html.Div(
                            id="row110",
                            children=[
                                html.H5(children="Category View Picker"),
                                dcc.Dropdown(
                                    id="bar_chart_view_picker",
                                    options=[{"label": col, "value": col} for col in
                                             self._data_handler.get_txt_categories],
                                    value=self._default_bar_pick
                                )
                            ]
                        ),
                        html.Div(
                            id="row111",
                            children=[
                                dcc.Loading(children=[dcc.Graph(id="scatter_kill_wound_selected_group_and_year",
                                                                className="graph_style")])
                            ]
                        )
                    ]
                )
            ]),

            html.Div(id="row2", children=[
                html.Div(
                    id="row20",
                    children=[
                        html.H5(children="Terrorism Group Selector"),
                        dcc.Dropdown(
                            id="group_selector",
                            options=[{"label": group, "value": group} for group in
                                     self._data_handler.get_data_frame_original["gname"].unique()],
                            value=self._default_group_list,
                            multi=True
                        ),
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
                    id="row21",
                    children=[
                        dcc.Loading(children=[dcc.Graph(id="map_year_attack_group_fig", className="graph_style")])
                    ]
                )
            ]),

            html.Div(
                id="row3",
                children=[
                    html.Div(
                        id="row30",
                        children=[
                            dcc.Loading(children=[dcc.Graph(id="line_year_attack_group_fig", className="graph_style")])
                        ]
                    ),
                    html.Div(
                        id="row31",
                        children=[
                            dcc.Loading(
                                children=[dcc.Graph(id="pie_year_target_selected_group_fig", className="graph_style")])
                        ]
                    )
                ]),
            html.Div(
                id="row4",
                children=[
                    html.Div(
                        id="row40",
                        children=[
                            dcc.Loading(children=[dcc.Graph(id="pca", className="graph_style")])
                        ]
                    )
                ]
            )
        ])

        @self._app.callback(
            dash.dependencies.Output("bar_year_attack_type_all_fig", "figure"),
            [dash.dependencies.Input("year_slider", "value"),
             dash.dependencies.Input("bar_chart_view_picker", "value")])
        def update_bar_year_attack_type_all_fig(year_range, selected_col):
            selected_col = selected_col if selected_col else self._default_bar_pick
            self._default_bar_pick = selected_col

            df = self._data_handler.get_data_frame_original
            df = DataHandler.trim_categories(data_frame=df, target_col=selected_col)
            df = df[df["iyear"].between(year_range[0], year_range[1], inclusive=True)]
            df = df.groupby(["iyear", selected_col]).size().reset_index(name="frequency")

            fig = px.bar(data_frame=df, x="iyear", y="frequency", color=selected_col,
                         custom_data=["iyear", selected_col], template="plotly_white")
            fig.update_layout(legend=self._default_legend_style,
                              autosize=True, title="Yearly Accumulated Attacks and Types")
            return fig

        @self._app.callback(
            dash.dependencies.Output("scatter_kill_wound_selected_group_and_year", "figure"),
            [dash.dependencies.Input("bar_year_attack_type_all_fig", "hoverData"),
             dash.dependencies.Input("bar_chart_view_picker", "value")])
        def update_scatter_kill_wound_selected_group_and_year(hovered_bar, selected_col):
            if hovered_bar and hovered_bar["points"] and hovered_bar["points"][0]["customdata"][1] in \
                    self._data_handler.get_data_frame_original[selected_col].unique():
                df_hovered_point = dict(zip(["iyear", selected_col], hovered_bar["points"][0]["customdata"]))
            else:
                df_hovered_point = {
                    "iyear": self._default_year_range[1],
                    selected_col: self._data_handler.get_data_frame_original[selected_col].value_counts().index[0]
                }

            df = self._data_handler.get_data_frame_original
            df = DataHandler.trim_categories(data_frame=df, target_col=selected_col)
            df = df[(df["iyear"] == df_hovered_point["iyear"]) & (df[selected_col] == df_hovered_point[selected_col])][
                ["natlty1_txt", "nkill", "nwound"]]
            df = DataHandler.trim_categories(data_frame=df, target_col="natlty1_txt")
            df["cases_by_" + df_hovered_point[selected_col]] = df["natlty1_txt"].map(df["natlty1_txt"].value_counts())

            fig = px.scatter(data_frame=df, x="nkill", y="nwound", color="natlty1_txt",
                             size=("cases_by_" + df_hovered_point[selected_col]))
            fig.update_layout(legend=self._default_legend_style, autosize=True,
                              title=("Detail view for " + df_hovered_point[selected_col]))
            return fig

        @self._app.callback(
            dash.dependencies.Output("map_year_attack_group_fig", "figure"),
            [dash.dependencies.Input("year_slider", "value"),
             dash.dependencies.Input("group_selector", "value")])
        def update_map_year_attack_group_fig(year_range, selected_group):
            df = self._data_handler.get_data_frame_original
            df = df[df["iyear"].between(year_range[0], year_range[1], inclusive=True)]
            df = df[df["gname"].isin(selected_group)]
            df = df.groupby(["gname", "latitude", "longitude"]).size().reset_index(name="frequency")

            fig = px.scatter_mapbox(data_frame=df, lat="latitude", lon="longitude", color="gname", size="frequency",
                                    zoom=2, custom_data=["gname", "latitude", "longitude"],
                                    color_continuous_scale=px.colors.cyclical.IceFire, mapbox_style="open-street-map",
                                    template="ggplot2")
            fig.update_layout(legend=self._default_legend_style, autosize=True,
                              title="Attacks Map by Terrorism Groups")
            return fig

        @self._app.callback(
            dash.dependencies.Output("line_year_attack_group_fig", "figure"),
            dash.dependencies.Output("pie_year_target_selected_group_fig", "figure"),
            [dash.dependencies.Input("year_slider", "value"),
             dash.dependencies.Input("map_year_attack_group_fig", "selectedData"),
             dash.dependencies.Input("group_selector", "value")])
        def update_line_pie_attack_target_selected_group(year_range, selected_points, selected_group):
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

            df_attack = df.groupby(["iyear", "gname"]).size().reset_index(name="frequency")
            df_target = df.groupby(["targtype1_txt"]).size().reset_index(name="frequency")

            fig_line = px.line(data_frame=df_attack, x="iyear", y="frequency", color="gname", template="plotly_white")
            fig_line.update_layout(legend=self._default_legend_style, autosize=True,
                                   title="Yearly Attacks by Terrorism Groups")

            fig_pie = px.pie(data_frame=df_target, values="frequency", names="targtype1_txt", template="ggplot2")
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(showlegend=False, autosize=True,
                                  title="Target Type by Selected Attacks")

            return fig_line, fig_pie

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

    def run_server(self):
        self._app.run_server(debug=True)


def main():
    path = "dataset/globalterrorismdb_0718dist.csv"
    visualizer = DataVisualizer.construct_from_csv(path=path)
    visualizer.run_server()


if __name__ == "__main__":
    main()
