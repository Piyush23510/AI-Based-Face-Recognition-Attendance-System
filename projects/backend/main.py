from fastapi import FastAPI
from attendance import add_attendance,auto_add_classes
from face_recognition import train_model

from schemas import (
    LoginSchema,
    RegisterSchema,
    AttendanceSchema,
    AbsenteeSchema,
    StudentSchema
)
from analytics import (
    calculate_percentage,
    get_students,
    get_absentees,
    get_today_attendance,
    add_student,
    get_classes,
    get_attendance,
    has_users

)

from auth import (
    login_user,
    register_user,
)


app = FastAPI()


@app.get("/")
def home():
    return {
        "message": "Attendance Backend Running"
    }


@app.post("/login")
def login(data: LoginSchema):

    result = login_user(
        data.username,
        data.password
    )

    if result:
        return result

    return {
        "success": False
    }


@app.post("/register")
def register(data: RegisterSchema):

    return register_user(
        data.username,
        data.password,
        data.role,
        data.roll
    )


@app.post("/attendance")
def attendance(data: AttendanceSchema):

    success = add_attendance(
        data.name,
        data.roll,
        data.subject
    )

    return {
        "success": success
    }


@app.get("/analytics")
def analytics():

    df = calculate_percentage()

    return df.to_dict(
        orient="records"
    )


@app.get("/students")
def students():

    df = get_students()

    return df.to_dict(
        orient="records"
    )


@app.post("/absentees")
def absentees(data: AbsenteeSchema):

    df = get_absentees(
        data.subject,
        data.selected_date
    )

    return df.to_dict(
        orient="records"
    )


@app.post("/train-model")
def train_face_model():

    return train_model()

@app.get("/today-attendance")
def today_attendance():

    df = get_today_attendance()

    return df.to_dict(
        orient="records"
    )

@app.get("/has-users")
def check_users():

    return {
        "has_users": has_users()
    }
@app.post("/students")
def create_student(data: StudentSchema):

    return add_student(
        data.name,
        data.roll
    )

@app.get("/classes")
def classes():

    df = get_classes()

    return df.to_dict(
        orient="records"
    )
@app.post("/auto-classes")
def create_classes():

    return auto_add_classes()

@app.get("/attendance-records")
def attendance_records():

    df = get_attendance()

    return df.to_dict(
        orient="records"
    )