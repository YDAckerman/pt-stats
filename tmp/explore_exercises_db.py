import sys
sys.path.append("/home/yoni/Projects/learn-stats/pt-stats")
from funs.db import create_conn, Queries
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np

TIME_STEP = .04

db_file = "/home/yoni/Projects/learn-stats/pt-stats/data/exercises.sqlite3"
conn = create_conn(db_file)
cur = conn.cursor()
qrys = Queries()

sens_data = pd.read_sql(qrys.select_time_labeled_sensor,
                        con=conn,
                        params={'subject': 1, 'exercise_type': 1, 'sensor_unit': 1})

sb.lineplot(data=sens_data,x="time_ind", y="acc_x", hue='execution_type')
