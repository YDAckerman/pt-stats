from src.result_handler import ResultHandler
import pickle
import numpy as np


class YoloDataCollection():

    def __init__(self, path_to_results):

        with open(path_to_results, 'rb') as res_file:
            self.results = pickle.load(res_file)
            self.handler = ResultHandler(self.results)
            self.pose_data = self.handler.kp_to_df()

    @staticmethod
    def get_kp_pairs(df, left_label, right_label):

        kp_pairs = df[df['label'] == left_label] \
            .merge(df[df['label'] == right_label],
                   left_on='frame',
                   right_on='frame',
                   suffixes=('_left', '_right'))

        return kp_pairs.copy()

    @staticmethod
    def get_kp_dists(df, new_column_name):

        df[new_column_name] = np.sqrt((df['X_left'] - df['X_right'])**2 +
                                      (df['Y_left'] - df['Y_right'])**2)
