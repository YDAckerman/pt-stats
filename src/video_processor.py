import pickle
import re
import os


class VideoProcessor():

    def __init__(self, model, save_path, conf=0.3):

        self.model = model
        self.save_path = save_path
        self.conf = conf

    def process_all_from(self, path_to_dir):
        vid_files = os.listdir(path_to_dir)
        print("Starting video processing...")

        for vf in vid_files:

            new_file_name = self.clean_filename(vf)
            res_path = self.save_path + "/" + new_file_name + ".pkl"

            print("\n")
            print(f"Starting to process {new_file_name} ...")
            results = self.process(path_to_dir + "/" + vf)

            print("Saving results...")
            with open(res_path, 'wb') as output:
                pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)

            print("Deleting results from RAM")
            del results

            print(f"{new_file_name} processing complete...")

        print("All videos processed. Stopping...")

    def process(self, path_to_data):
        return self.model(source=path_to_data,
                          show=False,
                          conf=self.conf,
                          show_labels=True)

    @staticmethod
    def clean_filename(filename):
        return re.sub(" ", "_", re.sub(".mp4", "", filename))
