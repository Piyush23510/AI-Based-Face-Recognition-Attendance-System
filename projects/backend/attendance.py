from database import get_conn
from datetime import datetime,date

def add_attendance(name, roll, subject):

    now = datetime.now()

    conn = get_conn()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT * FROM attendance
        WHERE roll=%s
        AND date=%s
        AND subject=%s
        """,
        (
            roll,
            str(now.date()),
            subject
        )
    )

    if cursor.fetchone() is None:

        cursor.execute(
            """
            INSERT INTO attendance
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                name,
                roll,
                subject,
                str(now.date()),
                now.strftime("%H:%M:%S")
            )
        )

        conn.commit()

        conn.close()

        return True

    conn.close()

    return False
timetable = {
    "Monday": ["AI", "ML"],
    "Tuesday": ["DBMS", "AI"],
    "Wednesday": ["ML"],
    "Thursday": ["AI", "DBMS"],
    "Friday": ["ML", "DBMS"],
    "Saturday": [],
    "Sunday": []
}


def auto_add_classes():

    today_day = datetime.today().strftime("%A")
    today_date = str(date.today())

    subjects_today = timetable.get(
        today_day,
        []
    )

    conn = get_conn()
    cursor = conn.cursor()

    for subject in subjects_today:

        cursor.execute(
            """
            SELECT *
            FROM classes
            WHERE subject=%s
            AND date=%s
            """,
            (
                subject,
                today_date
            )
        )

        if cursor.fetchone() is None:

            cursor.execute(
                """
                INSERT INTO classes
                VALUES (%s, %s)
                """,
                (
                    subject,
                    today_date
                )
            )

    conn.commit()
    conn.close()

    return {
        "success": True
    }