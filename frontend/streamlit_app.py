import os
import time
from datetime import date, datetime

import cv2
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st


API_URL = "https://ai-based-face-recognition-attendance-j7un.onrender.com"
SUBJECTS = ["AI", "ML", "DBMS"]
ADMIN_INVITE_CODE = os.getenv("ADMIN_INVITE_CODE", "A7xQ#29LmP@2026")
N_IMGS = 5
DATE_TODAY = str(date.today())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = (
    os.path.join(BASE_DIR, "static")
    if os.path.exists(os.path.join(BASE_DIR, "static"))
    else os.path.abspath(os.path.join(BASE_DIR, "..", "static"))
)

FACES_DIR = os.path.join(STATIC_DIR, "faces")
MODEL_PATH = os.path.join(STATIC_DIR, "face_recognition_model.pkl")
os.makedirs(FACES_DIR, exist_ok=True)

cascade_path = os.path.join(
    cv2.data.haarcascades,
    "haarcascade_frontalface_default.xml"
)
face_detector = cv2.CascadeClassifier(cascade_path)


st.set_page_config(
    page_title="Face Attendance",
    page_icon="📸",
    layout="wide"
)

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)


# -------------------------
# API helpers
# -------------------------
def api_get(endpoint):
    response = requests.get(
        f"{API_URL}{endpoint}",
        timeout=10
    )
    response.raise_for_status()
    return response.json()


def api_post(endpoint, data=None):
    response = requests.post(
        f"{API_URL}{endpoint}",
        json=data or {},
        timeout=20
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=10)
def get_students():
    return pd.DataFrame(api_get("/students"))


@st.cache_data(ttl=10)
def get_today_attendance():
    return pd.DataFrame(api_get("/today-attendance"))


@st.cache_data(ttl=10)
def get_all_attendance():
    return pd.DataFrame(api_get("/attendance-records"))


@st.cache_data(ttl=10)
def calculate_percentage():
    return pd.DataFrame(api_get("/analytics"))


@st.cache_data(ttl=10)
def get_classes():
    return pd.DataFrame(api_get("/classes"))


def get_absentees(subject, selected_date):
    return pd.DataFrame(
        api_post(
            "/absentees",
            {
                "subject": subject,
                "selected_date": selected_date
            }
        )
    )


def has_users():
    result = api_get("/has-users")
    return result.get("has_users", False)


def mark_attendance_api(name, roll, subject):
    return api_post(
        "/attendance",
        {
            "name": name,
            "roll": roll,
            "subject": subject
        }
    )


def train_model_api():
    return api_post("/train-model")


def auto_classes_api():
    return api_post("/auto-classes")


def add_student_api(name, roll):
    return api_post(
        "/students",
        {
            "name": name,
            "roll": roll
        }
    )


# -------------------------
# Common helpers
# -------------------------
def safe_roll(value):
    try:
        return int(value)
    except Exception:
        return None


def normalize_roll(value):
    try:
        return int(float(value))
    except Exception:
        return None


def current_user_roll():
    return normalize_roll(st.session_state.roll)


def is_student():
    return st.session_state.role == "user"


def filter_current_user(df):
    if df.empty or "roll" not in df.columns:
        return df

    df = df.copy()
    df["roll"] = df["roll"].apply(normalize_roll)
    return df[df["roll"] == current_user_roll()]


def show_dataframe(df):
    st.dataframe(df, use_container_width=True)


def show_centered_chart(fig):
    _, center, _ = st.columns([1, 3, 1])
    with center:
        st.plotly_chart(fig, use_container_width=True)


def clear_cache_after_success():
    st.cache_data.clear()


def stop_camera_and_rerun(delay=1):
    st.session_state.run_camera = False
    time.sleep(delay)
    st.rerun()


def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.2, 5)
    return faces if len(faces) > 0 else []


def decode_camera_image(picture):
    file_bytes = np.asarray(
        bytearray(picture.read()),
        dtype=np.uint8
    )
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def expected_role_for_portal(login_type):
    return "user" if login_type == "Student" else "admin"


def portal_name_for_role(role):
    return "Student" if role == "user" else "Admin"


def initialize_session():
    defaults = {
        "logged_in": False,
        "role": None,
        "roll": None,
        "run_camera": False,
        "camera_key": 0
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# -------------------------
# Auth section
# -------------------------
def handle_login(login_type):
    with st.form("login_form", clear_on_submit=True):
        st.markdown(f"#### {login_type} Login")

        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Login")

        if not submitted:
            return

        try:
            result = api_post(
                "/login",
                {
                    "username": username.strip(),
                    "password": password.strip()
                }
            )

            if not result.get("success"):
                st.error(result.get("message", "Invalid credentials!"))
                return

            role = result["role"]
            roll = result["roll"]
            expected_role = expected_role_for_portal(login_type)

            if role != expected_role:
                correct_portal = portal_name_for_role(role)
                st.error(f"You are a {correct_portal}. Please use the {correct_portal} portal.")
                return

            st.session_state.logged_in = True
            st.session_state.role = role
            st.session_state.roll = normalize_roll(roll)

            st.success(f"Logged in successfully as {role}!")
            time.sleep(1)
            st.rerun()

        except Exception as e:
            st.error(f"Login failed: {e}")


def handle_register(login_type):
    with st.form("register_form", clear_on_submit=True):
        st.markdown(f"#### Register as {login_type}")

        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        roll = None
        admin_key = None

        if login_type == "Student":
            roll = st.text_input("Roll Number")
        else:
            admin_key = st.text_input("Admin Invite Code (Required)", type="password")

        submitted = st.form_submit_button("Register")

        if not submitted:
            return

        if not new_user or not new_pass:
            st.warning("Please fill all details.")
            return

        if login_type == "Student" and not roll:
            st.warning("Please enter your roll number!")
            return

        if login_type == "Admin" and admin_key != ADMIN_INVITE_CODE:
            st.error("Invalid Admin Invite Code!")
            return

        db_role = expected_role_for_portal(login_type)
        roll_value = safe_roll(roll) if login_type == "Student" else 0

        if login_type == "Student" and roll_value is None:
            st.error("Roll number must be numeric!")
            return

        try:
            result = api_post(
                "/register",
                {
                    "username": new_user.strip(),
                    "password": new_pass.strip(),
                    "role": db_role,
                    "roll": roll_value
                }
            )

            if result.get("success"):
                st.success("Registration successful!")
                clear_cache_after_success()
            else:
                st.error(result.get("message", "Registration failed"))

        except Exception as e:
            st.error(f"Registration failed: {e}")


def show_auth_page():
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(
            "<h3 style='text-align: center;'>Login Portal</h3>",
            unsafe_allow_html=True
        )

        login_type = st.radio(
            "Choose Portal",
            ["Student", "Admin"],
            horizontal=True
        )

        tab1, tab2 = st.tabs(["Login", "Create Account"])

        with tab1:
            handle_login(login_type)

        with tab2:
            handle_register(login_type)

    st.stop()


# -------------------------
# Pages
# -------------------------
def show_dashboard():
    st.markdown(
        "<h1 style='text-align: center; color: #4CAF50;'>📊 Dashboard</h1>",
        unsafe_allow_html=True
    )

    df_students = get_students()

    if is_student():
        st.markdown("### 👤 My Profile")
        show_dataframe(filter_current_user(df_students))
    else:
        st.markdown("### 👨‍🎓 All Students")
        show_dataframe(df_students)

    df = get_today_attendance()

    if is_student():
        df = filter_current_user(df)
        st.markdown("### 📝 My Attendance Today")
    else:
        st.markdown("### 📝 Today's Attendance")

    if not df.empty:
        show_dataframe(df)
    else:
        st.info("No attendance marked yet today.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Classes Attended" if is_student() else "👥 Total Present", len(df))
    c2.metric(
        "📚 Subjects",
        df["subject"].nunique() if not df.empty and "subject" in df.columns else 0
    )
    c3.metric(
        "🕒 Last Entry",
        df["time"].iloc[-1] if not df.empty and "time" in df.columns else "--"
    )
    c4.metric("📅 Date", DATE_TODAY)

    if not df.empty:
        st.download_button(
            "Download Report",
            df.to_csv(index=False).encode(),
            f"attendance_{DATE_TODAY}.csv"
        )

    percentage_df = calculate_percentage()

    if not percentage_df.empty:
        low = percentage_df[percentage_df["attendance_%"] < 75]

        if not low.empty:
            st.warning("⚠️ Students below 75% attendance")
            show_dataframe(low)


def show_analytics():
    df = get_today_attendance()

    if is_student():
        df = filter_current_user(df)

    st.markdown("### 📝 Today's Log")

    if not df.empty:
        show_dataframe(df)
    else:
        st.info(
            "No classes attended today yet."
            if is_student()
            else "No attendance recorded yet today."
        )

    st.markdown("---")
    st.markdown("### 📈 Overall Attendance History")

    percentage_df = calculate_percentage()

    if not percentage_df.empty and is_student():
        percentage_df = filter_current_user(percentage_df)

    if not percentage_df.empty:
        fig = px.bar(
            percentage_df,
            x="subject" if is_student() else "name",
            y="attendance_%",
            color="subject",
            barmode="group",
            title="Overall Attendance %"
        )
        show_centered_chart(fig)
    else:
        st.info(
            "No historical attendance records found for you."
            if is_student()
            else "No historical attendance records found."
        )

    st.markdown("---")
    st.markdown("### 📊 Today's Insights")

    if df.empty:
        st.info("Not enough data today to generate insights.")
        return

    subject_count = df["subject"].value_counts().reset_index()
    subject_count.columns = ["subject", "count"]

    fig2 = px.pie(
        subject_count,
        values="count",
        names="subject",
        title="Today's Subject Distribution"
    )
    show_centered_chart(fig2)

    if not is_student():
        df_students = get_students()
        total_students = len(df_students)
        present = len(df)
        absent = max(total_students - present, 0)

        fig3 = px.pie(
            names=["Present", "Absent"],
            values=[present, absent],
            title="Today's Total Attendance Status"
        )
        show_centered_chart(fig3)


def handle_attendance_prediction(frame, subject):
    faces = extract_faces(frame)

    if len(faces) > 1:
        st.error("Multiple faces detected. Only one person allowed.")
        return

    if len(faces) == 0:
        st.error("No face detected")
        return

    model = joblib.load(MODEL_PATH)
    x, y, w, h = faces[0]
    face = cv2.resize(frame[y:y + h, x:x + w], (50, 50)).reshape(1, -1)

    prediction = model.predict(face)[0]
    distance, _ = model.kneighbors(face)

    if distance[0][0] > 4000:
        st.error("Unknown face detected")
        return

    try:
        student_name, roll_text = prediction.rsplit("_", 1)
        roll_value = int(roll_text)

        result = mark_attendance_api(student_name, roll_value, subject)

        if result.get("success"):
            st.success(f"Attendance marked for {student_name}")
            clear_cache_after_success()
        else:
            st.warning("Attendance already marked")

        stop_camera_and_rerun()

    except Exception as e:
        st.error(f"Attendance error: {e}")


def show_attendance_page():
    st.subheader("Camera Control")

    subject = st.selectbox("Select Subject", SUBJECTS)

    if st.button("Start Camera"):
        st.session_state.run_camera = True

        try:
            auto_classes_api()
            clear_cache_after_success()
        except Exception as e:
            st.warning(f"Could not auto-add class: {e}")

    try:
        users_available = has_users()
    except Exception as e:
        users_available = False
        st.error(f"Could not check users: {e}")

    if not users_available:
        st.error("❌ No users found. Please add a user first.")
        return

    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model not trained. Please add users and capture faces.")
        return

    if not st.session_state.run_camera:
        return

    picture = st.camera_input("Take Attendance")

    if picture is not None:
        with st.spinner("Processing attendance..."):
            frame = decode_camera_image(picture)
            handle_attendance_prediction(frame, subject)


def filter_attendance_by_date_subject(df, selected_date, subject_filter):
    filtered_df = df[df["date"] == selected_date]

    if subject_filter != "All":
        filtered_df = filtered_df[filtered_df["subject"] == subject_filter]

    if is_student():
        filtered_df = filter_current_user(filtered_df)

    return filtered_df


def class_was_conducted(subject_filter, selected_date):
    classes_df = get_classes()

    if classes_df.empty:
        return False

    check_class = classes_df[
        (classes_df["subject"] == subject_filter)
        & (classes_df["date"] == selected_date)
    ]

    return not check_class.empty


def show_absentee_status(subject_filter, selected_date):
    absent_df = get_absentees(subject_filter, selected_date)

    if not is_student():
        st.markdown("### Absentees")

        if absent_df.empty:
            st.success("No absentees")
        else:
            show_dataframe(absent_df)
            st.error(f"{len(absent_df)} students absent")

        return

    st.markdown("### My Status")

    if absent_df.empty:
        st.success("You were present!")
        return

    my_absent = filter_current_user(absent_df)

    if my_absent.empty:
        st.success("You were present!")
    else:
        st.error("You were absent for this class!")


def show_subjects_page():
    df = get_all_attendance()

    if df.empty:
        st.info("No attendance data available yet.")
        return

    selected_date = str(st.date_input("Select Date", datetime.today()))
    subject_filter = st.selectbox("Subject", ["All"] + SUBJECTS)

    filtered_df = filter_attendance_by_date_subject(df, selected_date, subject_filter)
    show_dataframe(filtered_df)

    if subject_filter == "All":
        return

    if not class_was_conducted(subject_filter, selected_date):
        st.warning("No class conducted on this date")
        return

    show_absentee_status(subject_filter, selected_date)


def save_student(name, user_id):
    if not name or not user_id:
        st.warning("Enter name and ID")
        return

    roll_val = safe_roll(user_id)

    if roll_val is None:
        st.error("Roll number must be numeric!")
        return

    try:
        result = add_student_api(name, roll_val)

        if result.get("success"):
            st.success("Student saved successfully!")
            clear_cache_after_success()
        else:
            st.error("Student could not be saved.")

    except Exception as e:
        st.error(f"Student save error: {e}")


def capture_face_image(name, user_id):
    if not name or not user_id:
        st.warning("Enter name and ID first")
        return

    if safe_roll(user_id) is None:
        st.error("Invalid roll number. Camera disabled.")
        return

    user_dir = os.path.join(FACES_DIR, f"{name}_{user_id}")
    os.makedirs(user_dir, exist_ok=True)

    existing_images = len(os.listdir(user_dir))

    if existing_images >= N_IMGS:
        st.success("Registration Complete")
        st.success("Model file created" if os.path.exists(MODEL_PATH) else "Model file not found")
        return

    picture = st.camera_input(
        f"Capture Face ({existing_images}/{N_IMGS})",
        key=f"camera_{st.session_state.camera_key}"
    )

    if picture is None:
        return

    frame = decode_camera_image(picture)
    faces = extract_faces(frame)

    if len(faces) == 0:
        st.error("No face detected!")
        return

    x, y, w, h = faces[0]
    face = frame[y:y + h, x:x + w]

    image_path = os.path.join(
        user_dir,
        f"{name}_{existing_images}.jpg"
    )
    cv2.imwrite(image_path, face)

    st.success(f"Image {existing_images + 1}/{N_IMGS} saved")
    st.session_state.camera_key += 1

    if existing_images + 1 >= N_IMGS:
        try:
            train_model_api()
            st.success("Face data captured & model trained!")
        except Exception as e:
            st.error(f"Model training failed: {e}")

    st.rerun()


def show_add_user_page():
    st.subheader("Register New User")

    name = st.text_input("Enter Name")
    user_id = st.text_input("Enter ID")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Save Student"):
            save_student(name, user_id)

    with col2:
        capture_face_image(name, user_id)


# -------------------------
# Main app
# -------------------------
initialize_session()

st.markdown("## Face Recognition Attendance System")
st.markdown("---")

if not st.session_state.logged_in:
    show_auth_page()

st.sidebar.title("📊 Navigation")

menu_options = (
    [
        "🏠 Admin Dashboard",
        "📸 Attendance",
        "📚 Subjects",
        "📊 Analytics",
        "➕ Add User"
    ]
    if st.session_state.role == "admin"
    else [
        "🏠 My Dashboard",
        "📚 My Subjects",
        "📊 My Analytics"
    ]
)

menu = st.sidebar.radio("Go to", menu_options)

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

if menu in ["🏠 Admin Dashboard", "🏠 My Dashboard"]:
    show_dashboard()
elif menu in ["📊 Analytics", "📊 My Analytics"]:
    show_analytics()
elif menu == "📸 Attendance":
    show_attendance_page()
elif menu in ["📚 Subjects", "📚 My Subjects"]:
    show_subjects_page()
elif menu == "➕ Add User" and st.session_state.role == "admin":
    show_add_user_page()
