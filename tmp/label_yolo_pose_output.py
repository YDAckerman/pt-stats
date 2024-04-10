import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sb
import pickle
import pandas as pd

LABELS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


with open('results_4_3_24.pkl', 'rb') as res_file:
    results = pickle.load(res_file)


kp = results[100].keypoints
xyn = pd.DataFrame(kp.xy.numpy()[0], columns=["X", "Y"])
xyn['label'] = LABELS
ax = sb.scatterplot(data=xyn, x='X', y='Y')


def label_points(df):
    for i, point in df.iterrows():
        if point['X'] != 0 and point['Y'] != 0:
            ax.text(point['X'] + .01,
                    point['Y'] + .01,
                    str(point['label']))

label_points(xyn)
