import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale


class DataHandler(object):

    def __init__(self, data_frame):
        self._data_frame_original = data_frame
        self._data_frame_last_update = None

        self._txt_cols = ["attacktype1_txt", "weaptype1_txt", "targtype1_txt", "gname"]
        self._numeric_cols = ["nperps", "nperpcap", "nkill", "nkillus", "nkillter", "nwound", "nwoundus", "nwoundte"]

        self.preprocess_data()

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

    @staticmethod
    def get_pca(data_frame, target, num_of_pc):
        x = StandardScaler().fit_transform(data_frame.loc[:, data_frame.columns != target])

        pca = PCA(n_components=num_of_pc)
        principal_components = pca.fit_transform(x)
        principal_df = pd.DataFrame(data=principal_components, columns=["P" + str(i + 1) for i in range(num_of_pc)])
        df_pca = pd.concat([principal_df, data_frame[target]], axis=1)

        return df_pca, pca.explained_variance_ratio_.sum()

    @staticmethod
    def get_clustering(data_frame, target, random_state):
        x = scale(data_frame.loc[:, data_frame.columns != target])
        y = data_frame[target]

        clustering = KMeans(n_clusters=len(y.unique()), random_state=random_state)
        clustering.fit(x)

        df_clustering = data_frame.copy()
        df_clustering[target] = clustering.labels_
        return df_clustering

    def apply_last_update(self):
        self._data_frame_original = self._data_frame_last_update.copy()

    def reset_last_update(self):
        self._data_frame_last_update = self._data_frame_original.copy()

    @staticmethod
    def get_top_categories(data_frame, target_cols, number_of_reserved=8):
        return list(data_frame[target_cols].value_counts().head(number_of_reserved).index)

    @staticmethod
    def trim_categories(data_frame, target_cols, designated_list=None, number_of_reserved=None):
        df = data_frame.copy()

        reserved_categories = designated_list if designated_list else data_frame[target_cols].value_counts().index
        df.loc[~data_frame[target_cols].isin(reserved_categories[:number_of_reserved]), target_cols] = "Other"

        return df

    @property
    def get_data_frame_original(self):
        return self._data_frame_original

    @property
    def get_data_frame_last_update(self):
        return self._data_frame_last_update

    @property
    def get_txt_columns(self):
        return self._txt_cols

    @property
    def get_numeric_columns(self):
        return self._numeric_cols


def unit_test():
    path = "dataset/globalterrorismdb_0718dist.csv"
    handler = DataHandler.construct_from_csv(path=path)
    print(DataHandler.trim_categories(handler.get_data_frame_original, "gname"))


if __name__ == '__main__':
    unit_test()
