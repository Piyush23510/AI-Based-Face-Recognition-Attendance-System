from pydantic import BaseModel

class LoginSchema(BaseModel):
    username: str
    password: str

class RegisterSchema(BaseModel):
    username: str
    password: str
    role: str
    roll: int

class AttendanceSchema(BaseModel):
    name: str
    subject: str
    roll : int

class AbsenteeSchema(BaseModel):
    subject: str
    selected_date: str

class StudentSchema(BaseModel):
    name: str
    roll: int