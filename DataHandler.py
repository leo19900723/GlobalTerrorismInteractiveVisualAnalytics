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
        self._numeric_cols = ["nperps", "nperpcap", "nkill", "nkillus", "nkillter", "nwound", "nwoundus", "nwoundte",
                              "propvalue", "nhostkid", "nhostkidus"]

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

    def get_data_frame_pca(self, tag, feature_cols):
        feature_and_tag = feature_cols + [tag]

        df = self._data_frame_original[feature_and_tag]
        df = df.dropna(subset=feature_and_tag).reset_index()

        x = StandardScaler().fit_transform(df[feature_cols])

        pca = PCA(n_components=3)
        principal_components = pca.fit_transform(x)
        principal_df = pd.DataFrame(data=principal_components, columns=["P1", "P2", "P3"])
        final_df = pd.concat([principal_df, df[tag]], axis=1)

        return final_df

    def get_data_frame_clustering(self, tag, feature_cols, random=5):
        feature_and_tag = feature_cols + [tag]

        df = self._data_frame_original[feature_and_tag]
        df = df.dropna(subset=feature_and_tag).reset_index()

        x = scale(df[feature_cols])
        y = df[tag]

        clustering = KMeans(n_clusters=len(y.unique()), random_state=random)
        clustering.fit(x)

        final_df = df
        final_df["predict"] = clustering.labels_
        return final_df

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
