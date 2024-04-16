from src.result_handler import ResultHandler as rh
import pickle


class YoloDataCollection():
  
    def __init__(self, path_to_results):

        with open(path_to_results, 'rb') as res_file:
            self.results = pickle.load(res_file)
            self.handler = rh(self.results)
            self.pose_data = self.handler.kp_to_df()
