import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class DataHandler(object):

    def __init__(self, data_frame):
        self._data_frame_original = data_frame
        self._data_frame_last_update = None
        self.preprocess_data()

        self._txt_cols = ["attacktype1_txt", "gname"]
        self._numeric_cols = ["nperps", "nperpcap", "nkill", "nkillus", "nkillter", "nwound", "nwoundus", "nwoundte",
                              "propvalue", "nhostkid", "nhostkidus", "nhours", "ndays", "nreleased"]

    @classmethod
    def construct_from_csv(cls, path):
        return DataHandler(data_frame=pd.read_csv(path, encoding="ISO-8859-1"))

    def preprocess_data(self):
        clean_list = [
            "iyear",
            "gname",
            "latitude",
            "longitude",
            "targtype1_txt",
            "weaptype1_txt",
            "attacktype1_txt"
        ]

        # Clean data
        self._data_frame_last_update = self._data_frame_original.replace(to_replace="Unknown", value=np.nan)
        self._data_frame_last_update = self._data_frame_last_update.dropna(subset=clean_list)
        self._data_frame_last_update = self._data_frame_last_update.reset_index()

        # Apply changes to original data_frame
        self.apply_last_update()

    def get_data_frame_pca(self, tag):

        self._data_frame_last_update = self._data_frame_original[self._numeric_cols]
        self._data_frame_last_update = self._data_frame_last_update.apply(lambda x: x.fillna(0))
        self._data_frame_last_update = pd.DataFrame(StandardScaler().fit_transform(self._data_frame_last_update),
                                                     columns=self._numeric_cols)

        pca = PCA(n_components=2)
        df_pca = pca.fit_transform(self._data_frame_last_update)
        df_pca = pd.DataFrame(data=df_pca, columns=["x", "y"])
        df_pca[tag] = self._data_frame_original[tag]

        return df_pca

    def apply_last_update(self):
        self._data_frame_original = self._data_frame_last_update.copy()

    def reset_last_update(self):
        self._data_frame_last_update = self._data_frame_original.copy()

    @staticmethod
    def trim_categories(data_frame, target_col, number_of_reserved=8):
        df = data_frame.copy()

        if len(data_frame[target_col].unique()) >= number_of_reserved:
            reserved_categories = data_frame[target_col].value_counts().head(number_of_reserved).index
            df.loc[~data_frame[target_col].isin(reserved_categories), target_col] = "Others"

        return df

    @property
    def get_data_frame_original(self):
        return self._data_frame_original

    @property
    def get_data_frame_last_update(self):
        return self._data_frame_last_update

    @property
    def get_txt_categories(self):
        return self._txt_cols


def unit_test():
    path = "dataset/globalterrorismdb_0718dist.csv"
    handler = DataHandler.construct_from_csv(path=path)
    print(DataHandler.trim_categories(handler.get_data_frame_original, "gname"))


if __name__ == '__main__':
    unit_test()
