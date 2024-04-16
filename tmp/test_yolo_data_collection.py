from src.yolo_data_collection import YoloDataCollection
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

ydc = YoloDataCollection('results_4_15_24.pkl')

walking_df = ydc.pose_data[0]

tmp = walking_df.merge(walking_df, left_on='frame', right_on='frame',
                       suffixes=('_left', '_right'))

gait_df = tmp[(tmp['label_left'] == 'left_ankle') &
              (tmp['label_right'] == 'right_ankle')]
