import pandas as pd
from database import get_conn
from datetime import date

def get_students():

    conn = get_conn()

    students_df = pd.read_sql_query(
        """
        SELECT username AS name, roll
        FROM users
        WHERE role='user'

        UNION

        SELECT name, roll
        FROM students
        """,
        conn
    )

    conn.close()

    return students_df


def calculate_percentage():

    conn = get_conn()

    classes_df = pd.read_sql_query(
        """
        SELECT subject,
               COUNT(*) AS total_classes
        FROM classes
        GROUP BY subject
        """,
        conn
    )

    attendance_df = pd.read_sql_query(
        """
        SELECT
            name,
            roll,
            subject,
            COUNT(*) AS classes_attended
        FROM attendance
        GROUP BY
            name,
            roll,
            subject
        """,
        conn
    )

    conn.close()

    if attendance_df.empty:
        return pd.DataFrame()

    merged_df = pd.merge(
        attendance_df,
        classes_df,
        on="subject",
        how="left"
    )

    merged_df["attendance_%"] = (
        merged_df["classes_attended"]
        / merged_df["total_classes"]
        * 100
    ).round(2)

    return merged_df


def get_absentees(subject, selected_date):

    conn = get_conn()

    students_df = pd.read_sql_query(
        """
        SELECT username AS name, roll
        FROM users
        WHERE role='user'

        UNION

        SELECT name, roll
        FROM students
        """,
        conn
    )

    present_df = pd.read_sql_query(
        """
        SELECT DISTINCT roll
        FROM attendance
        WHERE subject=%s
        AND date=%s
        """,
        conn,
        params=(subject, selected_date)
    )

    conn.close()

    absent_df = students_df[
        ~students_df["roll"].isin(
            present_df["roll"]
        )
    ]

    return absent_df

def get_today_attendance():

    conn = get_conn()

    today = str(date.today())

    df = pd.read_sql_query(
        """
        SELECT *
        FROM attendance
        WHERE date=%s
        """,
        conn,
        params=(today,)
    )

    conn.close()

    return df
def has_users():

    conn = get_conn()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM users
        WHERE role='user'
        """
    )

    count = cursor.fetchone()[0]

    conn.close()

    return count > 0
def add_student(name, roll):

    conn = get_conn()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO students (name, roll)
        VALUES (%s, %s)
        ON CONFLICT (roll)
        DO NOTHING
        """,
        (
            name,
            roll
        )
    )

    conn.commit()
    conn.close()

    return {
        "success": True
    }
def get_classes():

    conn = get_conn()

    df = pd.read_sql_query(
        """
        SELECT *
        FROM classes
        ORDER BY date DESC
        """,
        conn
    )

    conn.close()

    return df

import pandas as pd
from database import get_conn

def get_attendance():

    conn = get_conn()

    df = pd.read_sql_query(
        """
        SELECT *
        FROM attendance
        """,
        conn
    )

    conn.close()

    return df