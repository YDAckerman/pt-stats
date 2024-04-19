from src.result_handler import ResultHandler
import pickle


class YoloDataCollection():

    def __init__(self, path_to_results):

        with open(path_to_results, 'rb') as res_file:
            self.results = pickle.load(res_file)
            self.handler = ResultHandler(self.results)
            self.pose_data = self.handler.kp_to_df()

    @staticmethod
    def get_kp_pairs(df, left_label, right_label):

        all_pairs = df.merge(df,
                             left_on='frame',
                             right_on='frame',
                             suffixes=('_left', '_right'))

        return all_pairs[(all_pairs['label_left'] == left_label) &
                         (all_pairs['label_right'] == right_label)].copy()
