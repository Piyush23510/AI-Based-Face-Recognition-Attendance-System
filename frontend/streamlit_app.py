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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(os.path.join(BASE_DIR, "static")):
    STATIC_DIR = os.path.join(BASE_DIR, "static")
else:
    STATIC_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "static"))

FACES_DIR = os.path.join(STATIC_DIR, "faces")
MODEL_PATH = os.path.join(STATIC_DIR, "face_recognition_model.pkl")

os.makedirs(FACES_DIR, exist_ok=True)

nimgs = 5
datetoday = str(date.today())

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


def extract_faces(img):
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        1.2,
        5
    )

    return faces if len(faces) > 0 else []


@st.cache_data(ttl=10)
def get_students():
    return pd.DataFrame(
        api_get("/students")
    )


@st.cache_data(ttl=10)
def get_today_attendance():
    return pd.DataFrame(
        api_get("/today-attendance")
    )


@st.cache_data(ttl=10)
def get_all_attendance():
    return pd.DataFrame(
        api_get("/attendance-records")
    )


@st.cache_data(ttl=10)
def calculate_percentage():
    return pd.DataFrame(
        api_get("/analytics")
    )


@st.cache_data(ttl=10)
def get_classes():
    return pd.DataFrame(
        api_get("/classes")
    )


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


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.roll = None

if "backend_started" not in st.session_state:
    try:
        auto_classes_api()
        st.session_state.backend_started = True
    except Exception:
        st.session_state.backend_started = False


st.markdown("## Face Recognition Attendance System")
st.markdown("---")


if not st.session_state.logged_in:

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

        tab1, tab2 = st.tabs(
            ["Login", "Create Account"]
        )

        with tab1:
            with st.form(
                "login_form",
                clear_on_submit=True
            ):

                st.markdown(
                    f"#### {login_type} Login"
                )

                username = st.text_input(
                    "Username",
                    key="login_username"
                )

                password = st.text_input(
                    "Password",
                    type="password",
                    key="login_password"
                )

                submitted = st.form_submit_button(
                    "Login"
                )

                if submitted:

                    try:
                        result = api_post(
                            "/login",
                            {
                                "username": username.strip(),
                                "password": password.strip()
                            }
                        )

                        if result.get("success"):

                            role = result["role"]
                            roll = result["roll"]

                            if (
                                login_type.lower() == "student"
                                and role == "admin"
                            ):
                                st.error(
                                    "You are an Admin. Please use the Admin portal."
                                )

                            elif (
                                login_type.lower() == "admin"
                                and role == "user"
                            ):
                                st.error(
                                    "You are a Student. Please use the Student portal."
                                )

                            else:
                                st.session_state.logged_in = True
                                st.session_state.role = role
                                st.session_state.roll = normalize_roll(roll)

                                st.success(
                                    f"Logged in successfully as {role}!"
                                )

                                time.sleep(1)
                                st.rerun()

                        else:
                            st.error(
                                "Invalid credentials!"
                            )

                    except Exception as e:
                        st.error(
                            f"Login failed: {e}"
                        )

        with tab2:
            with st.form(
                "register_form",
                clear_on_submit=True
            ):

                st.markdown(
                    f"#### Register as {login_type}"
                )

                new_user = st.text_input(
                    "New Username"
                )

                new_pass = st.text_input(
                    "New Password",
                    type="password"
                )

                roll = None
                admin_key = None
                admin_invite_code = "A7xQ#29LmP@2026"

                if login_type == "Student":
                    roll = st.text_input(
                        "Roll Number"
                    )
                else:
                    admin_key = st.text_input(
                        "Admin Invite Code (Required)",
                        type="password"
                    )

                submitted = st.form_submit_button(
                    "Register"
                )

                if submitted:

                    if not new_user or not new_pass:
                        st.warning(
                            "Please fill all details."
                        )

                    elif login_type == "Student" and not roll:
                        st.warning(
                            "Please enter your roll number!"
                        )

                    elif (
                        login_type == "Admin"
                        and admin_key != admin_invite_code
                    ):
                        st.error(
                            "Invalid Admin Invite Code!"
                        )

                    else:
                        db_role = (
                            "user"
                            if login_type == "Student"
                            else "admin"
                        )

                        roll_value = (
                            safe_roll(roll)
                            if login_type == "Student"
                            else 0
                        )

                        try:
                            result = api_post(
                                "/register",
                                {
                                    "username": new_user,
                                    "password": new_pass,
                                    "role": db_role,
                                    "roll": roll_value
                                }
                            )

                            if result.get("success"):
                                st.success(
                                    "Registration successful!"
                                )
                                st.cache_data.clear()

                            else:
                                st.error(
                                    result.get(
                                        "message",
                                        "Registration failed"
                                    )
                                )

                        except Exception as e:
                            st.error(
                                f"Registration failed: {e}"
                            )

    st.stop()


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

menu = st.sidebar.radio(
    "Go to",
    menu_options
)

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()


if menu in ["🏠 Admin Dashboard", "🏠 My Dashboard"]:

    st.markdown(
        "<h1 style='text-align: center; color: #4CAF50;'>📊 Dashboard</h1>",
        unsafe_allow_html=True
    )

    df_students = get_students()

    if st.session_state.role == "admin":

        st.markdown("### 👨‍🎓 All Students")
        st.dataframe(
            df_students,
            use_container_width=True
        )

    else:

        st.markdown("### 👤 My Profile")

        if not df_students.empty:
            df_students["roll"] = df_students["roll"].apply(
                normalize_roll
            )

            my_info = df_students[
                df_students["roll"]
                == normalize_roll(st.session_state.roll)
            ]

            st.dataframe(
                my_info,
                use_container_width=True
            )

    df = get_today_attendance()

    if st.session_state.role == "user":

        if not df.empty:
            df["roll"] = df["roll"].apply(
                normalize_roll
            )

            df = df[
                df["roll"]
                == normalize_roll(st.session_state.roll)
            ]

        st.markdown("### 📝 My Attendance Today")

    else:
        st.markdown("### 📝 Today's Attendance")

    if not df.empty:
        st.dataframe(
            df,
            use_container_width=True
        )
    else:
        st.info(
            "No attendance marked yet today."
        )

    c1, c2, c3, c4 = st.columns(4)

    if st.session_state.role == "user":
        c1.metric(
            "👥 Classes Attended",
            len(df)
        )
    else:
        c1.metric(
            "👥 Total Present",
            len(df)
        )

    c2.metric(
        "📚 Subjects",
        df["subject"].nunique()
        if not df.empty
        and "subject" in df.columns
        else 0
    )

    c3.metric(
        "🕒 Last Entry",
        df["time"].iloc[-1]
        if not df.empty
        and "time" in df.columns
        else "--"
    )

    c4.metric(
        "📅 Date",
        datetoday
    )

    if not df.empty:
        st.download_button(
            "Download Report",
            df.to_csv(index=False).encode(),
            f"attendance_{datetoday}.csv"
        )

    percentage_df = calculate_percentage()

    if not percentage_df.empty:
        low = percentage_df[
            percentage_df["attendance_%"] < 75
        ]

        if not low.empty:
            st.warning(
                "⚠️ Students below 75% attendance"
            )
            st.dataframe(
                low,
                use_container_width=True
            )


elif menu in ["📊 Analytics", "📊 My Analytics"]:

    df = get_today_attendance()

    if st.session_state.role == "user":

        if not df.empty:
            df["roll"] = df["roll"].apply(
                normalize_roll
            )

            df = df[
                df["roll"]
                == normalize_roll(st.session_state.roll)
            ]

    st.markdown("### 📝 Today's Log")

    if not df.empty:
        st.dataframe(
            df,
            use_container_width=True
        )
    else:
        st.info(
            "No classes attended today yet."
            if st.session_state.role == "user"
            else "No attendance recorded yet today."
        )

    st.markdown("---")
    st.markdown("### 📈 Overall Attendance History")

    percentage_df = calculate_percentage()

    if not percentage_df.empty:

        if st.session_state.role == "user":

            percentage_df["roll"] = percentage_df["roll"].apply(
                normalize_roll
            )

            percentage_df = percentage_df[
                percentage_df["roll"]
                == normalize_roll(st.session_state.roll)
            ]

        if not percentage_df.empty:

            col_left, col_center, col_right = st.columns(
                [1, 3, 1]
            )

            with col_center:

                fig = px.bar(
                    percentage_df,
                    x=(
                        "subject"
                        if st.session_state.role == "user"
                        else "name"
                    ),
                    y="attendance_%",
                    color="subject",
                    barmode="group",
                    title="Overall Attendance %"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        else:
            st.info(
                "No historical attendance records found for you."
            )

    else:
        st.info(
            "No historical attendance records found."
        )

    st.markdown("---")
    st.markdown("### 📊 Today's Insights")

    if not df.empty:

        col_left, col_center, col_right = st.columns(
            [1, 3, 1]
        )

        with col_center:

            subject_count = df[
                "subject"
            ].value_counts().reset_index()

            subject_count.columns = [
                "subject",
                "count"
            ]

            fig2 = px.pie(
                subject_count,
                values="count",
                names="subject",
                title="Today's Subject Distribution"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

            if st.session_state.role == "admin":

                df_students = get_students()

                total_students = len(df_students)
                present = len(df)
                absent = max(
                    total_students - present,
                    0
                )

                fig3 = px.pie(
                    names=[
                        "Present",
                        "Absent"
                    ],
                    values=[
                        present,
                        absent
                    ],
                    title="Today's Total Attendance Status"
                )

                st.plotly_chart(
                    fig3,
                    use_container_width=True
                )

    else:
        st.info(
            "Not enough data today to generate insights."
        )


elif menu == "📸 Attendance":

    st.subheader("Camera Control")

    subject = st.selectbox(
        "Select Subject",
        ["AI", "ML", "DBMS"]
    )

    if "run_camera" not in st.session_state:
        st.session_state.run_camera = False

    if st.button("Start Camera"):
        st.session_state.run_camera = True

        try:
            auto_classes_api()
            st.cache_data.clear()
        except Exception as e:
            st.warning(
                f"Could not auto-add class: {e}"
            )

    try:
        users_available = has_users()
    except Exception as e:
        users_available = False
        st.error(
            f"Could not check users: {e}"
        )

    if not users_available:
        st.error(
            "❌ No users found. Please add a user first."
        )

    elif not os.path.exists(MODEL_PATH):
        st.error(
            "❌ Model not trained. Please add users and capture faces."
        )

    else:

        if st.session_state.run_camera:

            picture = st.camera_input(
                "Take Attendance"
            )

            if picture is not None:

                with st.spinner(
                    "Processing attendance..."
                ):

                    file_bytes = np.asarray(
                        bytearray(picture.read()),
                        dtype=np.uint8
                    )

                    frame = cv2.imdecode(
                        file_bytes,
                        cv2.IMREAD_COLOR
                    )

                    faces = extract_faces(frame)

                    if len(faces) == 1:

                        model = joblib.load(
                            MODEL_PATH
                        )

                        x, y, w, h = faces[0]

                        face = cv2.resize(
                            frame[y:y + h, x:x + w],
                            (50, 50)
                        ).reshape(1, -1)

                        prediction = model.predict(face)[0]

                        distance, _ = model.kneighbors(face)

                        if distance[0][0] > 4000:

                            st.error(
                                "Unknown face detected"
                            )

                        else:

                            try:
                                student_name, roll_text = prediction.rsplit(
                                    "_",
                                    1
                                )

                                roll_value = int(
                                    roll_text
                                )

                                result = mark_attendance_api(
                                    student_name,
                                    roll_value,
                                    subject
                                )
                                if result.get("success"):
                                  st.success(
                                      f"Attendance marked for {student_name}"
                                  )

                                  st.cache_data.clear()

                                  st.session_state.run_camera = False

                                  time.sleep(1)

                                  st.rerun()
                                else:
                                  st.warning(
                                      "Attendance already marked"
                                  )

                                  st.session_state.run_camera = False

                                  time.sleep(1)

                                  st.rerun()                              

                            except Exception as e:
                                st.error(
                                    f"Attendance error: {e}"
                                )

                    elif len(faces) > 1:
                        st.error(
                            "Multiple faces detected. Only one person allowed."
                        )

                    else:
                        st.error(
                            "No face detected"
                        )


elif menu in ["📚 Subjects", "📚 My Subjects"]:

    df = get_all_attendance()

    if not df.empty:

        selected_date = str(
            st.date_input(
                "Select Date",
                datetime.today()
            )
        )

        subject_filter = st.selectbox(
            "Subject",
            ["All", "AI", "ML", "DBMS"]
        )

        filtered_df = df[
            df["date"] == selected_date
        ]

        if subject_filter != "All":
            filtered_df = filtered_df[
                filtered_df["subject"]
                == subject_filter
            ]

        if st.session_state.role == "user":

            filtered_df["roll"] = filtered_df["roll"].apply(
                normalize_roll
            )

            filtered_df = filtered_df[
                filtered_df["roll"]
                == normalize_roll(st.session_state.roll)
            ]

        st.dataframe(
            filtered_df,
            use_container_width=True
        )

        if subject_filter != "All":

            classes_df = get_classes()

            if not classes_df.empty:

                check_class = classes_df[
                    (
                        classes_df["subject"]
                        == subject_filter
                    )
                    & (
                        classes_df["date"]
                        == selected_date
                    )
                ]

            else:
                check_class = pd.DataFrame()

            if check_class.empty:

                st.warning(
                    "No class conducted on this date"
                )

            else:

                absent_df = get_absentees(
                    subject_filter,
                    selected_date
                )

                if st.session_state.role == "admin":

                    st.markdown("### Absentees")

                    if not absent_df.empty:
                        st.dataframe(
                            absent_df,
                            use_container_width=True
                        )

                        st.error(
                            f"{len(absent_df)} students absent"
                        )

                    else:
                        st.success(
                            "No absentees"
                        )

                else:

                    st.markdown("### My Status")

                    if not absent_df.empty:

                        absent_df["roll"] = absent_df["roll"].apply(
                            normalize_roll
                        )

                        my_absent = absent_df[
                            absent_df["roll"]
                            == normalize_roll(st.session_state.roll)
                        ]

                        if not my_absent.empty:
                            st.error(
                                "You were absent for this class!"
                            )

                        else:
                            st.success(
                                "You were present!"
                            )

                    else:
                        st.success(
                            "You were present!"
                        )

    else:
        st.info(
            "No attendance data available yet."
        )


elif menu == "➕ Add User" and st.session_state.role == "admin":

    st.subheader("Register New User")

    name = st.text_input(
        "Enter Name"
    )

    user_id = st.text_input(
        "Enter ID"
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button("Save Student"):

            if not name or not user_id:
                st.warning(
                    "Enter name and ID"
                )

            else:
                roll_val = safe_roll(user_id)

                if roll_val is None:

                    st.error(
                        "Roll number must be numeric!"
                    )

                else:

                    try:
                        result = add_student_api(
                            name,
                            roll_val
                        )

                        if result.get("success"):
                            st.success(
                                "Student saved successfully!"
                            )
                            st.cache_data.clear()

                        else:
                            st.error(
                                "Student could not be saved."
                            )

                    except Exception as e:
                        st.error(
                            f"Student save error: {e}"
                        )

    with col2:

        if "camera_key" not in st.session_state:
            st.session_state.camera_key = 0

        if not name or not user_id:
            st.warning(
                "Enter name and ID first"
            )

        elif safe_roll(user_id) is None:
            st.error(
                "Invalid roll number. Camera disabled."
            )

        else:

            user_dir = os.path.join(
                FACES_DIR,
                f"{name}_{user_id}"
            )

            os.makedirs(
                user_dir,
                exist_ok=True
            )

            existing_images = len(
                os.listdir(user_dir)
            )

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

                        st.error(
                            "No face detected!"
                        )

                    else:

                        x, y, w, h = faces[0]

                        face = frame[
                            y:y + h,
                            x:x + w
                        ]

                        image_path = os.path.join(
                            user_dir,
                            f"{name}_{existing_images}.jpg"
                        )

                        cv2.imwrite(
                            image_path,
                            face
                        )

                        st.success(
                            f"Image {existing_images + 1}/{nimgs} saved"
                        )

                        st.session_state.camera_key += 1

                        if existing_images + 1 >= nimgs:

                            try:
                                train_model_api()

                                st.success(
                                    "Face data captured & model trained!"
                                )

                            except Exception as e:
                                st.error(
                                    f"Model training failed: {e}"
                                )

                        st.rerun()

            else:

                st.success(
                    "Registration Complete"
                )

                if os.path.exists(MODEL_PATH):
                    st.success(
                        "Model file created"
                    )
                else:
                    st.error(
                        "Model file not found"
                    )
