import sys
sys.path.append("/home/yoni/Projects/learn-stats/pt-stats")
import pandas as pd
from funs.db import create_conn, Queries


def extract_transform_load(file_name, **kwargs):

    motion_dt = extract_motion_data(file_name)
    format_column_names(motion_dt)
    add_meta_data(motion_dt, **kwargs)
    load_motion_data(motion_dt, **kwargs)

    pass


def load_motion_data(motion_dt, **kwargs):

    try:
        motion_dt.to_sql(kwargs.get('destination_table'),
                         con=kwargs.get('conn'),
                         if_exists="append", index=False)
    except Exception as e:
        print(e)

    pass


def extract_motion_data(file_name):

    return pd.read_csv(file_name, sep=";", header=0)


def format_column_names(motion_dt):

    if 'time index' in motion_dt.columns:
        motion_dt.rename(columns={'time index': 'time_ind'},
                         inplace=True)

    if 'execution type' in motion_dt.columns:
        motion_dt.rename(columns={'execution type': 'execution_type'},
                         inplace=True)

    pass


def add_meta_data(motion_dt, **kwargs):

    motion_dt['subject'] = pd.Series([kwargs.get('_s', None)] *
                                     motion_dt.shape[0])
    motion_dt['exercise_type'] = pd.Series([kwargs.get('_e', None)] *
                                           motion_dt.shape[0])
    motion_dt['sensor_unit'] = pd.Series([kwargs.get('_u', None)] *
                                         motion_dt.shape[0])

    pass


if __name__ == '__main__':

    db_file = "/home/yoni/Projects/learn-stats/pt-stats/data/exercises.sqlite3"
    conn = create_conn(db_file)
    cur = conn.cursor()
    qrys = Queries()

    cur.execute(qrys.drop_template_table)
    cur.execute(qrys.drop_time_table)

    conn.commit()

    for _s in range(1, 6):
        for _e in range(1, 9):
            for _u in range(1, 6):

                base_dir = "/home/yoni/Projects/learn-stats/pt-stats/"
                times_data_dir = f"data/exercises/s{_s}/e{_e}/"
                sens_data_dir = f"data/exercises/s{_s}/e{_e}/u{_u}/"

                kwargs = {'_s': _s, '_e': _e, '_u': _u, 'conn': conn}

                table_to_file = {'session': sens_data_dir + 'test.txt',
                                 'template': sens_data_dir + 'template_session.txt',
                                 'times': times_data_dir + 'template_times.txt'}

                for tbl, fname in table_to_file.items():
                    kwargs['destination_table'] = tbl
                    extract_transform_load(fname, **kwargs)

    conn.commit()
    conn.close()
