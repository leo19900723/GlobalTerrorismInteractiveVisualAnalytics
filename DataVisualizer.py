import dash
import dash_html_components as html
import plotly.express as px
import dash_core_components as dcc

from DataHandler import DataHandler


class DataVisualizer(object):

    def __init__(self, app, data_handler):
        self._app = app
        self._data_handler = data_handler

        self.set_layout()

    @classmethod
    def construct_from_data_handler(cls, path):
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

        return DataVisualizer(
            app=dash.Dash(__name__, external_stylesheets=external_stylesheets),
            data_handler=DataHandler.construct_from_csv(path)
        )

    def set_layout(self):
        self._data_handler.top10_group_from_2000()
        # see https://plotly.com/python/px-arguments/ for more options
        fig = px.line(data_frame=self._data_handler.get_dataframe_last_update, x="iyear", y="frequency", color="gname")

        self._app.layout = html.Div(children=[
            html.H1(children="Hello Dash"),
            html.Div(children="Dash: A web application framework for Python."),
            dcc.Graph(
                id='example-graph',
                figure=fig
            )
        ])

    def run_server(self):
        self._app.run_server(debug=True)


def main():
    path = "dataset/globalterrorismdb_0718dist.csv"
    visualizer = DataVisualizer.construct_from_data_handler(path=path)
    visualizer.run_server()


if __name__ == "__main__":
    main()
