import pandas as pd
import numpy as np


class DataHandler(object):

    def __init__(self, dataframe):
        self._dataframe_original = dataframe
        self._dataframe_last_update = None
        self.clean_data()

    @classmethod
    def construct_from_csv(cls, path):
        return DataHandler(dataframe=pd.read_csv(path, encoding="ISO-8859-1"))

    def clean_data(self):
        self._dataframe_last_update = self._dataframe_original[self._dataframe_original["gname"] != "Unknown"].dropna(subset=["gname"])

        self.apply_last_update()

    def top10_group_from_2000(self):
        self._dataframe_last_update = self._dataframe_original[self._dataframe_original["iyear"] >= 2000].dropna(subset=["iyear"])
        df_2000_top10_list = self._dataframe_last_update["gname"].value_counts().head(10).index

        self._dataframe_last_update = self._dataframe_last_update.groupby(["iyear", "gname"]).size().reset_index(name="frequency")
        self._dataframe_last_update = self._dataframe_last_update[self._dataframe_last_update["gname"].isin(df_2000_top10_list)]

    def apply_last_update(self):
        self._dataframe_original = self._dataframe_last_update.copy()

    def reset_last_update(self):
        self._dataframe_last_update = self._dataframe_original.copy()

    @property
    def get_dataframe_original(self):
        return self._dataframe_original

    @property
    def get_dataframe_last_update(self):
        return self._dataframe_last_update


def unit_test():
    path = "dataset/globalterrorismdb_0718dist.csv"
    handler = DataHandler.construct_from_csv(path=path)
    print(handler.get_dataframe_original)


if __name__ == '__main__':
    unit_test()
