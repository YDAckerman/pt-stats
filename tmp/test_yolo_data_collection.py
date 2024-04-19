from src.yolo_data_collection import YoloDataCollection
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np

ydc = YoloDataCollection('results_4_15_24.pkl')

# ###########################################################
# Stride Length
# ###########################################################

lat_wlk_df = ydc.pose_data[1]

sb.scatterplot(data=lat_wlk_df[lat_wlk_df['frame'] == 0], x='X', y='Y')
plt.show()

gait_df = ydc.get_kp_pairs(lat_wlk_df, 'left_ankle', 'right_ankle')

gait_df['ankle_dist'] = np.sqrt((gait_df['X_left'] - gait_df['X_right'])**2 +
                                (gait_df['Y_left'] - gait_df['Y_right'])**2)

sb.lineplot(data=gait_df, x='frame', y='ankle_dist')
plt.show()

# ###########################################################
# Hip angle
# ###########################################################

frnt_wlk_df = ydc.pose_data[0]

sb.scatterplot(data=frnt_wlk_df[frnt_wlk_df['frame'] == 0], x='X', y='Y')
plt.show()

hip_df = ydc.get_kp_pairs(frnt_wlk_df, 'left_hip', 'right_hip')


def angle_to_x_deg(row):
    dx = row['X_left'] - row['X_right']
    dy = row['Y_left'] - row['Y_right']
    norm = np.sqrt(dx**2 + dy**2)
    ux = dx / norm
    uy = dy / norm
    rad = np.arccos(np.clip(np.dot((ux, uy), (1.0, 0.0)), -1.0, 1.0))
    return rad * 180.0 / np.pi


hip_df['hip_angle'] = hip_df.apply(lambda row: angle_to_x_deg(row), axis=1)

sb.lineplot(data=hip_df, x='frame', y='hip_angle')
plt.show()
