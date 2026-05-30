import bcrypt
from database import get_conn


def register_user(username, password, role, roll):
    conn = get_conn()
    cursor = conn.cursor()

    username = username.strip()
    password = password.strip()

    # Check if username already exists
    cursor.execute(
        "SELECT * FROM users WHERE username=%s",
        (username,)
    )

    if cursor.fetchone():
        conn.close()
        return {
            "success": False,
            "message": "Username already exists"
        }

    # Hash password
    hashed_password = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")

    # Insert user into database
    cursor.execute(
        """
        INSERT INTO users (username, password, role, roll)
        VALUES (%s, %s, %s, %s)
        """,
        (username, hashed_password, role, roll)
    )

    conn.commit()
    conn.close()

    return {
        "success": True
    }


def login_user(username, password):
    conn = get_conn()
    cursor = conn.cursor()

    # Get stored password and user details
    cursor.execute(
        """
        SELECT password, role, roll
        FROM users
        WHERE username=%s
        """,
        (username,)
    )

    result = cursor.fetchone()

    conn.close()

    if not result:
        return return {
            "success": False,
            "message": "User not registered. Please register first."
        }


    stored_password, role, roll = result

    # Check password
    if bcrypt.checkpw(
        password.encode("utf-8"),
        stored_password.encode("utf-8")
    ):
        return {
            "success": True,
            "role": role,
            "roll": roll
        }

    return None
