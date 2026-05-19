import streamlit as st
import cv2
import os
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import pandas as pd
import joblib
import sqlite3
import time
import plotly.express as px

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Face Attendance", layout="wide")

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

nimgs = 5
datetoday = str(date.today())

# FIX 1: Use cv2.data.haarcascades to prevent "missing haarcascade_frontalface_default.xml" errors
cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_detector = cv2.CascadeClassifier(cascade_path)
os.makedirs('static/faces', exist_ok=True)

def register_user(username, password, role, roll):
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    username = username.strip()
    password = password.strip()

    try:
        cursor.execute(
            "INSERT INTO users (username, password, role, roll) VALUES (?, ?, ?, ?)",
            (username, password, role, roll)
        )
        conn.commit()
    except Exception as e:
        st.error(f"Registration error: {e}")

    conn.close()

def login_user(username, password):
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    username = username.strip()
    password = password.strip()

    cursor.execute(
        "SELECT role, roll FROM users WHERE username=? AND password=?",
        (username, password)
    )

    result = cursor.fetchone()
    conn.close()
    return result

# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        name TEXT,
        roll INTEGER,
        subject TEXT,
        date TEXT,
        time TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS students (
        name TEXT,
        roll INTEGER PRIMARY KEY
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS classes (
        subject TEXT,
        date TEXT
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
       username TEXT PRIMARY KEY,
       password TEXT,
       role TEXT,
       roll INTEGER
    )
    """)

    conn.commit()
    conn.close()

init_db()

# ---------------- TIMETABLE ----------------
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

    subjects_today = timetable.get(today_day, [])

    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    for subject in subjects_today:
        cursor.execute("""
            SELECT * FROM classes 
            WHERE subject=? AND date=?
        """, (subject, today_date))

        if cursor.fetchone() is None:
            cursor.execute("""
                INSERT INTO classes VALUES (?, ?)
            """, (subject, today_date))

    conn.commit()
    conn.close()
    
auto_add_classes()

# ---------------- FUNCTIONS ----------------
def normalize_roll(value):
    try:
        return int(float(value))
    except:
        return None

def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.2, 5)
    return faces if len(faces) > 0 else []

def train_model():
    faces = []
    labels = []

    for user in os.listdir('static/faces'):

        user_path = os.path.join('static/faces', user)

        if not os.path.isdir(user_path):
            continue

        print("Training for:", user)

        for imgname in os.listdir(user_path):

            img_path = os.path.join(user_path, imgname)

            img = cv2.imread(img_path)

            if img is None:
                continue

            resized_face = cv2.resize(img, (50, 50)).ravel()

            faces.append(resized_face)
            labels.append(user)

    print("Total Faces:", len(faces))
    print("Labels:", set(labels))

    if len(faces) > 0:

        n_neighbors = min(5, len(faces))

        knn = KNeighborsClassifier(n_neighbors=n_neighbors)

        knn.fit(np.array(faces), labels)

        joblib.dump(knn, 'static/face_recognition_model.pkl')

        print("Model trained successfully")

def add_attendance(name, subject):
    # FIX 3: Robust string split in case of multiple underscores
    parts = name.rsplit('_', 1)
    if len(parts) == 2:
        username, userid = parts
    else:
        username, userid = name, "0"

    now = datetime.now()

    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM attendance 
        WHERE roll=? AND date=? AND subject=?
    """, (int(userid), str(now.date()), subject))

    if cursor.fetchone() is None:

        print("Detected User:", username)
        print("Detected Roll:", userid)

        cursor.execute("""
            INSERT INTO attendance VALUES (?, ?, ?, ?, ?)
        """, (username, int(userid), subject, str(now.date()), now.strftime("%H:%M:%S")))
        conn.commit()
        conn.close()
        return True

    conn.close()
    return False

def get_today_attendance():
    conn = sqlite3.connect('attendance.db')

    df = pd.read_sql_query(
        "SELECT * FROM attendance WHERE date=?",
        conn,
        params=(datetoday,)
    )

    conn.close()
    return df
def calculate_percentage():
    conn = sqlite3.connect('attendance.db')

    classes_df = pd.read_sql_query(
        "SELECT subject, COUNT(*) as total_classes FROM classes GROUP BY subject",
        conn
    )

    attendance_df = pd.read_sql_query(
        """
        SELECT name, roll, subject, COUNT(*) as classes_attended
        FROM attendance
        GROUP BY name, roll, subject
        """,
        conn
    )

    conn.close()

    if attendance_df.empty:
        return pd.DataFrame()

    merged = pd.merge(attendance_df, classes_df, on="subject", how="left")

    merged["attendance_%"] = (
        merged["classes_attended"] / merged["total_classes"] * 100
    ).round(2)

    return merged

def get_absentees(subject, selected_date):
    conn = sqlite3.connect('attendance.db')

    students = pd.read_sql_query("""
        SELECT username AS name, roll FROM users WHERE role='user'
        UNION
        SELECT name, roll FROM students
    """, conn)

    present = pd.read_sql_query("""
        SELECT DISTINCT roll 
        FROM attendance 
        WHERE subject=? AND date=?
    """, conn, params=(subject, selected_date))

    conn.close()

    if students.empty:
        return pd.DataFrame()

    absent = students[~students["roll"].isin(present["roll"])]
    return absent

# FIX 4: Replace deprecated VideoTransformerBase with VideoProcessorBase
def get_processor_factory(subject):
    class FaceRecognitionProcessor(VideoProcessorBase):
        def __init__(self):
            self.model = joblib.load('static/face_recognition_model.pkl')
            self.marked = set()
            self.subject = subject

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            faces = extract_faces(img)

            for (x, y, w, h) in faces[:1]:
                face_img = cv2.resize(img[y:y+h, x:x+w], (50, 50)).reshape(1, -1)

                name = self.model.predict(face_img)[0]

                print("Predicted:", name)

                if name not in self.marked and "_" in name:
                    if add_attendance(name, self.subject):
                        self.marked.add(name)

                cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(img, name, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
    return FaceRecognitionProcessor

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None    

# ---------------- UI ----------------
st.markdown("## Face Recognition Attendance System")
st.markdown("---")

if not st.session_state.logged_in:

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("<h3 style='text-align: center;'>Login Portal</h3>", unsafe_allow_html=True)
        
        login_type = st.radio("Choose Portal", ["Student", "Admin"], horizontal=True)
        
        tab1, tab2 = st.tabs(["Login", "Create Account"])

        with tab1:
            with st.form("login_form"):
                st.markdown(f"#### {login_type} Login")
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")

                submitted = st.form_submit_button("Login")

                if submitted:
                    result = login_user(username, password)

                    if result:
                        role, roll = result
                        if login_type.lower() == "student" and role == "admin":
                            st.error("You are an Admin. Please use the Admin portal.")
                        elif login_type.lower() == "admin" and role == "user":
                            st.error("You are a Student. Please use the Student portal.")
                        else:
                            st.session_state.logged_in = True
                            st.session_state.role = role
                            st.session_state.roll = int(roll) if roll is not None else None
                            st.success(f"Logged in successfully as {role}!")
                            time.sleep(1)
                            st.rerun()
                    else:
                        st.error("Invalid credentials!")

        with tab2:
            with st.form("register_form"):
                st.markdown(f"#### Register as {login_type}")
                new_user = st.text_input("New Username")
                new_pass = st.text_input("New Password", type="password")
                
                roll = None
                admin_key = None
                
                if login_type == "Student":
                    roll = st.text_input("Roll Number")
                else:
                    admin_key = st.text_input("Admin Invite Code (Required)", type="password")
                    st.caption("Hint: The admin invite code is 'admin123'")

                if st.form_submit_button("Register"):
                    if new_user and new_pass:
                        if login_type == "Student" and not roll:
                            st.warning("Please enter your roll number!")
                        elif login_type == "Admin" and admin_key != "admin123":
                            st.error("Invalid Admin Invite Code!")
                        else:
                            db_role = "user" if login_type == "Student" else "admin"
                            register_user(
                                new_user,
                                new_pass,
                                db_role,
                                int(roll) if roll else None
                            )
                            st.success(f"{login_type} account created! You can now login.")
                    else:
                        st.warning("Please fill all details.")

    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Navigation")

if st.session_state.role == "admin":
    menu_options = [
        "🏠 Admin Dashboard",
        "📸 Attendance",
        "📚 Subjects",
        "📊 Analytics",
        "➕ Add User"
    ]
else:
    menu_options = [
        "🏠 My Dashboard",
        "📚 My Subjects",
        "📊 My Analytics"
    ]

menu = st.sidebar.radio("Go to", menu_options)

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.role = None
    st.rerun()

# ---------------- HOME ----------------
if menu in ["🏠 Admin Dashboard", "🏠 My Dashboard"]:

    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📊 Dashboard</h1>", unsafe_allow_html=True)

    conn = sqlite3.connect('attendance.db')
    df_students = pd.read_sql_query("""
        SELECT username AS name, roll FROM users WHERE role='user'
        UNION
        SELECT name, roll FROM students
    """, conn)
    conn.close()

    if st.session_state.role == "admin":
        st.markdown("### 👨‍🎓 All Students")
        st.dataframe(df_students)
    else:
        st.markdown("### 👤 My Profile")
        if not df_students.empty:
            df_students["roll"] = df_students["roll"].apply(normalize_roll)
            my_info = df_students[
                df_students["roll"] == normalize_roll(st.session_state.roll)
            ]
            st.dataframe(my_info)
    
    df = get_today_attendance()

    if st.session_state.role == "user":
     if not df.empty:
        df["roll"] = df["roll"].apply(normalize_roll)
        df = df[
          df["roll"] == normalize_roll(st.session_state.roll)
      ]
     st.markdown("### 📝 My Attendance Today")
    else:
       st.markdown("### 📝 Today's Attendance")

    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No attendance marked yet today.")

    c1, c2, c3, c4 = st.columns(4)
    if st.session_state.role == "user":
        c1.metric("👥 Classes Attended", len(df))
    else:
        c1.metric("👥 Total Present", len(df))
    c2.metric("📚 Subjects", df["subject"].nunique() if not df.empty else 0)
    c3.metric("🕒 Last Entry", df["time"].iloc[-1] if not df.empty else "--")
    c4.metric("📅 Date", datetoday)

    if not df.empty:
        st.download_button(
            "Download Report",
            df.to_csv(index=False).encode(),
            f"attendance_{datetoday}.csv"
        )

    percentage_df = calculate_percentage()
    if not percentage_df.empty:
        low = percentage_df[percentage_df["attendance_%"] < 75]

        if not low.empty:
            st.warning("⚠️ Students below 75% attendance")
            st.dataframe(low)

elif menu in ["📊 Analytics", "📊 My Analytics"]:
    
    df = get_today_attendance()

    if st.session_state.role == "user":
      if not df.empty:
        df["roll"] = df["roll"].apply(normalize_roll)
        df = df[
            df["roll"] == normalize_roll(st.session_state.roll)
        ]
    st.markdown("### 📝 Today's Log")
    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No classes attended today yet." if st.session_state.role == "user" else "No attendance recorded yet today.")

    st.markdown("---")
    
    st.markdown("### 📈 Overall Attendance History")
    percentage_df = calculate_percentage()

    if not percentage_df.empty:
        if st.session_state.role == "user":
            percentage_df["roll"] = percentage_df["roll"].apply(normalize_roll)
            percentage_df = percentage_df[
            percentage_df["roll"] == normalize_roll(st.session_state.roll)
            ]

        if not percentage_df.empty:
            col_left, col_center, col_right = st.columns([1, 3, 1])
            with col_center:
                fig = px.bar(
                    percentage_df,
                    x="subject" if st.session_state.role == "user" else "name",
                    y="attendance_%",
                    color="subject",
                    barmode="group",
                    title="Overall Attendance %"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No historical attendance records found for you.")
    else:
        st.info("No historical attendance records found.")

    st.markdown("---")
    st.markdown("### 📊 Today's Insights")

    if not df.empty:
        col_left, col_center, col_right = st.columns([1, 3, 1])

        with col_center:
            subject_count = df["subject"].value_counts().reset_index()
            subject_count.columns = ["subject", "count"]

            fig2 = px.pie(
                subject_count,
                values="count",
                names="subject",
                title="Today's Subject Distribution"
            )
            st.plotly_chart(fig2, use_container_width=True)

            if st.session_state.role == "admin":
                conn = sqlite3.connect('attendance.db')
                df_students = pd.read_sql_query("""
                    SELECT username AS name, roll FROM users WHERE role='user'
                    UNION
                    SELECT name, roll FROM students
                """, conn)
                conn.close()

                total_students = len(df_students)
                present = len(df)
                absent = max(total_students - present, 0)

                fig3 = px.pie(
                    names=["Present", "Absent"],
                    values=[present, absent],
                    title="Today's Total Attendance Status"
                )
                st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Not enough data today to generate insights.")

# ---------------- START CAMERA ----------------
elif menu == "📸 Attendance":

    st.subheader("Camera Control")

    subject = st.selectbox("Select Subject", ["AI", "ML", "DBMS"])

    if "run_camera" not in st.session_state:
        st.session_state.run_camera = False

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start Camera"):
            st.session_state.run_camera = True

            conn = sqlite3.connect('attendance.db')
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM classes 
                WHERE subject=? AND date=?
            """, (subject, str(date.today())))

            if not cursor.fetchone():
                cursor.execute("""
                    INSERT INTO classes VALUES (?, ?)
                """, (subject, str(date.today())))

            conn.commit()
            conn.close()

    with col2:
        if st.button("Stop Camera"):
            st.session_state.run_camera = False
            st.success("✅ Attendance marked successfully!")

    if not os.path.exists('static/face_recognition_model.pkl'):
        st.error("Model not trained yet! Please add a user first.")
    else:
        if st.session_state.run_camera:
            webrtc_streamer(
                key="attendance",
                video_processor_factory=get_processor_factory(subject) # FIX 5: Uses new processor factory
            )

# ---------------- SUBJECT TABLES ----------------
elif menu in ["📚 Subjects", "📚 My Subjects"]:

    conn = sqlite3.connect('attendance.db')
    df = pd.read_sql_query("SELECT * FROM attendance", conn)
    conn.close()

    if not df.empty:

        selected_date = str(st.date_input("Select Date", datetime.today()))
        subject_filter = st.selectbox("Subject", ["All", "AI", "ML", "DBMS"])

        filtered_df = df[df["date"] == selected_date]

        if subject_filter != "All":
           filtered_df = filtered_df[filtered_df["subject"] == subject_filter]

        if st.session_state.role == "user":
           filtered_df["roll"] = filtered_df["roll"].apply(normalize_roll)
           filtered_df = filtered_df[
               filtered_df["roll"] == normalize_roll(st.session_state.roll)
           ]

        st.dataframe(filtered_df, use_container_width=True)

        if subject_filter != "All":
            conn = sqlite3.connect('attendance.db')
            check_class = pd.read_sql_query(
                "SELECT * FROM classes WHERE subject=? AND date=?",
                conn,
                params=(subject_filter, selected_date)
            )
            conn.close()

            if check_class.empty:
                st.warning("⚠️ No class conducted on this date")
            else:
                absent_df = get_absentees(subject_filter, selected_date)
                
                if st.session_state.role == "admin":
                    st.markdown("### 🚫 Absentees")
                    if not absent_df.empty:
                        st.dataframe(absent_df, use_container_width=True)
                        st.error(f"{len(absent_df)} students absent")
                    else:
                        st.success("No absentees 🎉")
                else:
                    st.markdown("### 🚫 My Status")
                    if not absent_df.empty:
                        absent_df["roll"] = absent_df["roll"].apply(normalize_roll)
                        my_absent = absent_df[
                            absent_df["roll"] == normalize_roll(st.session_state.roll)
                        ]
                        if not my_absent.empty:
                            st.error("You were absent for this class!")
                        else:
                            st.success("You were present! 🎉")
                    else:
                        st.success("You were present! 🎉")
    else:
        st.info("No attendance data available yet.")

# ---------------- ADD USER ----------------
elif menu == "➕ Add User" and st.session_state.role == "admin":

    st.subheader("Register New User")

    name = st.text_input("Enter Name")
    user_id = st.text_input("Enter ID")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("💾 Save Student"):
            if name and user_id:
                conn = sqlite3.connect('attendance.db')
                cursor = conn.cursor()

                cursor.execute(
                    "INSERT OR IGNORE INTO students VALUES (?, ?)",
                    (name, int(user_id))
                )

                conn.commit()
                conn.close()

                st.success("✅ Student saved successfully!")
            else:
                st.warning("⚠️ Enter name and ID")
    with col2:

      if "camera_key" not in st.session_state:
        st.session_state.camera_key = 0     
      if not name or not user_id:

        st.button("📸 Capture Face", disabled=True)
        st.warning("⚠️ Enter name and ID first")

      else:

        user_dir = f"static/faces/{name}_{user_id}"
        os.makedirs(user_dir, exist_ok=True)

        existing_images = len(os.listdir(user_dir))

        # Hide camera after enough images collected
        if existing_images < nimgs:

            picture = st.camera_input(
                f"Capture Face ({existing_images}/{nimgs})",
                key=f"camera_{st.session_state.camera_key}"
            )

            if picture is not None:

                file_bytes = np.asarray(
                    bytearray(picture.read()),
                    dtype=np.uint8
                )

                frame = cv2.imdecode(
                    file_bytes,
                    cv2.IMREAD_COLOR
                )

                faces = extract_faces(frame)

                if len(faces) == 0:

                    st.error("❌ No face detected!")

                else:

                    x, y, w, h = faces[0]

                    face = frame[y:y+h, x:x+w]

                    cv2.imwrite(
                        f"{user_dir}/{name}_{existing_images}.jpg",
                        face
                    )

                    st.success(
                        f"✅ Image {existing_images + 1}/{nimgs} saved"
                    )

                    # Refresh camera for next image
                    st.session_state.camera_key += 1

                    if existing_images + 1 >= nimgs:

                        train_model()

                        st.success(
                            "🎉 Face data captured & model trained!"
                        )

                    st.rerun()

        else:

            st.success("✅ Registration Complete")

            if os.path.exists(
                "static/face_recognition_model.pkl"
            ):
                st.success("✅ Model file created")
            else:
                st.error("❌ Model file not found")
