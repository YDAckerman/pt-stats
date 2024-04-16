import pickle
import pandas as pd


LABELS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


with open('results_4_15_24.pkl', 'rb') as res_file:
    results = pickle.load(res_file)


def keypoints_to_dataframe(i, keypoints):
    xyn = pd.DataFrame(keypoints.xyn.numpy()[0], columns=["X", "Y"])
    xyn['label'] = LABELS
    xyn['frame'] = [i] * len(LABELS)
    return xyn


kp_df_list = [keypoints_to_dataframe(i, frame.keypoints)
              for i, frame in enumerate(results)]

kp_df = pd.concat(kp_df_list)
