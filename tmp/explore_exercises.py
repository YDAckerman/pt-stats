import os
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# for a single file (sensor)
subject, exercise, sensor = [1,1,1]
base_dir = "/home/yoni/Projects/learn-stats/pt-stats/"
times_data_dir = f"data/exercises/s{subject}/e{exercise}/"
sens_data_dir = f"data/exercises/s{subject}/e{exercise}/u{sensor}/"

# data from a single sensor:
ex_data = pd.read_csv(sens_data_dir + "template_session.txt", sep = ";", header=0)

# the exercise is repeated 3 times, given here: 
ex_times = pd.read_csv(times_data_dir + "template_times.txt", sep = ";", header=0)

# add in time in seconds, as per dataset description
ex_data['time_sec'] = .04*(ex_data['time index']-1)

# crude integration approximation
for axis in ['x', 'y', 'z']:
    ex_data[f'vel_{axis}'] = .04 * ex_data[f'acc_{axis}']
    ex_data[f'pos_{axis}'] = .04 * ex_data[f'vel_{axis}']
    
# exercise labeling function
def label_exercise(row):
    filt = ex_times.loc[(ex_times['start'] <= row['time index']) &
                         (ex_times['end'] >= row['time index'])]
    if filt.empty:
        return 0
    return filt.iloc[0]['execution type']

ex_data['exercise_no'] = ex_data.apply(label_exercise, axis=1)

sb.lineplot(data=ex_data.loc[ex_data.exercise_no == 1],
            x="time index", y="acc_x", hue='exercise_no')

def plot_ex_no(n) :
    axes = plot.axes(projection="3d")
    axes.plot3D(ex_data.loc[ex_data.exercise_no == n]['acc_x'],
                ex_data.loc[ex_data.exercise_no == n]['acc_y'],
                ex_data.loc[ex_data.exercise_no == n]['acc_z'])
    plt.tight_layout()
    plt.show()
