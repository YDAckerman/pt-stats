from src.yolo_data_collection import YoloDataCollection
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np

ydc = YoloDataCollection('results_4_15_24.pkl')

FOREARM_LEN_INCHES = 11

# ###########################################################
# Stride Length
# ###########################################################

lat_wlk_df = ydc.pose_data[1]

forearm_df = ydc.get_kp_pairs(lat_wlk_df, 'right_elbow', 'right_wrist')
ydc.get_kp_dists(forearm_df, 'forearm_dist')
forearm_dist = forearm_df['forearm_dist'].mean()

gait_df = ydc.get_kp_pairs(lat_wlk_df, 'left_ankle', 'right_ankle')
ydc.get_kp_dists(gait_df, 'ankle_dist')

gait_df['ankle_dist_inches'] = gait_df['ankle_dist'] * FOREARM_LEN_INCHES / forearm_dist

ax = sb.lineplot(data=gait_df, x='frame', y='ankle_dist_inches')
ax.set(xlabel='Frame',
       ylabel='Stride Length (Inches)',
       title='Stride Length')
plt.show()

# ###########################################################
# Hip angle
# ###########################################################

frnt_wlk_df = ydc.pose_data[0]
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

ax = sb.lineplot(data=hip_df, x='frame', y='hip_angle')
ax.set(xlabel='Frame',
       ylabel='Hip Angle (Degrees)',
       title='Hip Angle')
plt.show()
