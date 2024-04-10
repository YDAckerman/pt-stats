import sqlite3
from sqlite3 import Error

def create_conn(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print("db connection created")
    except Error as e:
        print(e)
    finally:
        if conn:
            return conn


class Queries:

    drop_template_table = """
    DROP TABLE IF EXISTS template;
    """

    drop_session_table = """
    DROP TABLE IF EXISTS session;
    """

    drop_time_table = """
    DROP TABLE IF EXISTS time;
    """

    select_time_labeled_sensor = """
    SELECT * FROM 
    session s JOIN time t
    ON s.subject = t.subject AND 
       s.exercise_type = t.exercise_type AND 
       s.sensor_unit = t.sensor_unit AND 
       s.time_ind >= t.start AND 
       s.time_ind <= t.end
    WHERE s.subject = :subject AND 
          s.exercise_type = :exercise_type AND
          s.sensor_unit = :sensor_unit;
    """
