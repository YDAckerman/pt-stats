import os
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import dtw as dtw
from whittaker_eilers import WhittakerSmoother
from ultralytics import YOLO
from src.video_processor import VideoProcessor
from src.yolo_data_collection import YoloDataCollection
from src.funs.extract_strides import extract_strides

# #############################################
# Analysis Constants
# #############################################

BASE_PATH = os.path.abspath(".")
MODEL = YOLO(BASE_PATH + '/models/yolov8m-pose.pt')
RESULTS_SAVE_PATH = BASE_PATH + '/yolo_results/anim_ref_res'
VID_PATH = BASE_PATH + "/raw_video/animation_reference"
FOREARM_LEN_INCHES = 11

# #############################################
# process all videos
# #############################################

VideoProcessor(MODEL, RESULTS_SAVE_PATH).process_all_from(VID_PATH)

# #############################################
# extract desired keypoints
# #############################################
kp_dfs = []

for i, path in enumerate(os.listdir(RESULTS_SAVE_PATH)):
    ydc = YoloDataCollection(RESULTS_SAVE_PATH + "/" + path)
    lat_wlk_df = ydc.pose_data[1]
    forearm_df = ydc.get_kp_pairs(lat_wlk_df, 'right_elbow', 'right_wrist')
    ydc.get_kp_dists(forearm_df, 'forearm_dist')
    forearm_dist = forearm_df['forearm_dist'].mean()
    gait_df = ydc.get_kp_pairs(lat_wlk_df, 'left_ankle', 'right_ankle')
    ydc.get_kp_dists(gait_df, 'ankle_dist')
    gait_df['ankle_dist_inches'] = gait_df['ankle_dist'] * FOREARM_LEN_INCHES / forearm_dist
    model_ref = ' '.join(path.split("_")[0:4])
    gait_df['model_reference'] = [model_ref] * gait_df.shape[0]
    gait_df['model_reference_num'] = [i] * gait_df.shape[0]
    kp_dfs.append(gait_df)

all_model_df = pd.concat(kp_dfs)

# counts
all_model_df.model_reference_num.unique()
all_model_df.model_reference.unique()
all_model_df.groupby("model_reference").count()

# #############################################
# plot Standard Male
# #############################################
ax = sb.lineplot(data=all_model_df[all_model_df.model_reference_num == 3],
                 x='frame',
                 y='ankle_dist_inches',
                 hue='model_reference')
ax.set(xlabel='Frame',
       ylabel='Stride Length (Inches)',
       title='Stride Length')
plt.show()

# #############################################
# Plot Standard Female vs Standard Male
# #############################################

ax = sb.lineplot(data=all_model_df[(all_model_df.model_reference_num == 0) |
                                   (all_model_df.model_reference_num == 3)],
                 x='frame',
                 y='ankle_dist_inches',
                 hue='model_reference')
ax.set(xlabel='Frame',
       ylabel='Stride Length (Inches)',
       title='Stride Length')
plt.show()

# #############################################
# plot all curves
# #############################################
ax = sb.lineplot(data=all_model_df, x='frame',
                 y='ankle_dist_inches',
                 hue='model_reference')
ax.set(xlabel='Frame',
       ylabel='Stride Length (Inches)',
       title='Stride Length')
plt.show()

# #############################################
# Whittaker Eilers Smoothing
# #############################################

# ref: https://towardsdatascience.com/the-perfect-way-to-smooth-your-noisy-data-4f3fe6b44440

std_fem = all_model_df[(all_model_df.model_reference_num == 2)].copy()

whittaker_smoother = WhittakerSmoother(
    lmbda=100, order=2, data_length=std_fem.shape[0]
)

std_fem['whit_smooth'] = whittaker_smoother.smooth(std_fem.ankle_dist_inches.values)
ax = sb.lineplot(data=std_fem,
                 x='frame',
                 y='whit_smooth',
                 hue='model_reference')
ax.set(xlabel='Frame',
       ylabel='Smoothed Stride Length (Inches)',
       title='Stride Length')
plt.show()

# #############################################
# Extracting Strides
# #############################################

peaks = [i for i, v in enumerate(std_fem.whit_smooth)
         if (i > 0 and i < std_fem.shape[0] - 1) and
         v > std_fem.whit_smooth.values[i-1] and
         v > std_fem.whit_smooth.values[i+1] and
         v > np.quantile(std_fem.whit_smooth.values, .9)]

troughs = [i for i, v in enumerate(std_fem.whit_smooth)
           if (i > 0 and i < std_fem.shape[0] - 1) and
           v < std_fem.whit_smooth.values[i-1] and
           v < std_fem.whit_smooth.values[i+1] and
           i > peaks[0] and i < peaks[-1]]

ax = sb.lineplot(data=std_fem.loc[troughs[1]:troughs[5]],
                 x='frame',
                 y='ankle_dist_inches',
                 hue='model_reference')
ax.set(xlabel='Frame',
       ylabel='Smoothed Stride Length (Inches)',
       title='Stride Length')
plt.show()


# #############################################
# Batch Extract Strides
# #############################################

stride_dfs = []
for i in all_model_df.model_reference_num.unique():
    df_slice_cp = all_model_df[all_model_df.model_reference_num == i].copy()
    strides_i = extract_strides(df_slice_cp, 'ankle_dist_inches', 1, 5)
    stride_dfs.append(strides_i)
    del df_slice_cp

stride_df = pd.concat(stride_dfs)


ax = sb.lineplot(data=stride_df, x='frame',
                 y='whit_smooth',
                 hue='model_reference')
ax.set(xlabel='Frame',
       ylabel='Stride Length (Inches)',
       title='Stride Length')
plt.show()


# #############################################
# Dynamic Time Warping
# #############################################

# threeway
alignment = dtw.dtw(stride_df[stride_df.model_reference_num == 0]
                    .whit_smooth,
                    stride_df[stride_df.model_reference_num == 2]
                    .whit_smooth,
                    keep_internals=True
                    )

alignment.plot(type="threeway")
plt.show()

# two way
dtw.dtw(stride_df[stride_df.model_reference_num == 0].ankle_dist_inches,
        stride_df[stride_df.model_reference_num == 2].ankle_dist_inches,
        keep_internals=True,
        step_pattern=dtw.rabinerJuangStepPattern(6, "c")
        ).plot(type="twoway", offset=-2)
plt.show()
