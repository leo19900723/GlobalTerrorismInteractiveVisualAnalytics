import pandas as pd
import numpy as np


class DataHandler(object):

    def __init__(self, dataframe):
        self._dataframe_original = dataframe
        self._dataframe_last_update = None
        self.preprocess_data()

    @classmethod
    def construct_from_csv(cls, path):
        return DataHandler(dataframe=pd.read_csv(path, encoding="ISO-8859-1"))

    def preprocess_data(self):
        clean_list = [
            "iyear",
            "gname",
            "latitude",
            "longitude",
            "targtype1_txt",
            "weaptype1_txt"
        ]

        # Clean data
        self._dataframe_last_update = self._dataframe_original.replace(to_replace="Unknown", value=np.nan)
        self._dataframe_last_update = self._dataframe_last_update.dropna(subset=clean_list)

        # Apply changes to original dataframe
        self.apply_last_update()

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
