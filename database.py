
import json
from datetime import datetime

from sqlalchemy import (
    create_engine, Column, Integer, String,
    Text, DateTime, inspect, text
)
from sqlalchemy.orm import declarative_base, sessionmaker


# DATABASE CONFIGURATION
DATABASE_URL = "sqlite:///student_system.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False
)

Base = declarative_base()


# USER MODEL
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    salt = Column(String(100), nullable=False)
    role = Column(String(30), default="faculty", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# STUDENT MODEL
class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    input_data = Column(Text)
    predicted_risk = Column(String(100))
    created_by = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)


# AUDIT LOG MODEL
class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100))
    action = Column(String(100))
    detail = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)


# DATABASE MIGRATION
def migrate_database():

    Base.metadata.create_all(bind=engine)

    migrations = {
        "users": {
            "password_hash": "VARCHAR(256) DEFAULT ''",
            "salt": "VARCHAR(100) DEFAULT ''",
            "role": "VARCHAR(30) DEFAULT 'faculty'",
            "created_at": "DATETIME"
        },
        "students": {
            "input_data": "TEXT",
            "predicted_risk": "VARCHAR(100)",
            "created_by": "VARCHAR(100)",
            "created_at": "DATETIME"
        },
        "audit_logs": {
            "username": "VARCHAR(100)",
            "action": "VARCHAR(100)",
            "detail": "TEXT",
            "timestamp": "DATETIME"
        }
    }

    inspector = inspect(engine)
    tables = inspector.get_table_names()

    for table, columns in migrations.items():

        if table not in tables:
            continue

        existing = {
            col["name"]
            for col in inspector.get_columns(table)
        }

        with engine.begin() as connection:

            for column, definition in columns.items():

                if column not in existing:

                    connection.execute(
                        text(
                            f'ALTER TABLE "{table}" '
                            f'ADD COLUMN "{column}" {definition}'
                        )
                    )

                    print(f"Added {table}.{column}")


def init_db():
    migrate_database()
    print("Database initialized successfully.")


# STUDENT FUNCTIONS
def save_student(db, input_data, prediction, username):

    if hasattr(input_data, "to_dict"):
        input_data = input_data.to_dict()

    student = Student(
        input_data=json.dumps(input_data, default=str),
        predicted_risk=str(prediction),
        created_by=username
    )

    db.add(student)
    db.flush()

    return student.id


def get_students(db):

    records = db.query(Student).order_by(
        Student.id.desc()
    ).all()

    result = []

    for student in records:

        try:
            data = json.loads(student.input_data or "{}")
        except (ValueError, TypeError):
            data = {}

        if not isinstance(data, dict):
            data = {"student_data": data}

        result.append({
            "id": student.id,
            **data,
            "predicted_risk": student.predicted_risk,
            "created_by": student.created_by,
            "created_at": str(student.created_at or "")
        })

    return result


def delete_student(db, student_id):

    student = db.query(Student).filter(
        Student.id == int(student_id)
    ).first()

    if student:
        db.delete(student)
        db.flush()
        return True

    return False


# USER MANAGEMENT FUNCTIONS
def get_all_users(db):

    users = db.query(User).order_by(User.id).all()

    return [
        {
            "id": user.id,
            "username": user.username,
            "role": user.role,
            "created_at": str(user.created_at or "")
        }
        for user in users
    ]


def delete_user(db, user_id):

    user = db.query(User).filter(
        User.id == int(user_id)
    ).first()

    if user:
        db.delete(user)
        db.flush()
        return True

    return False


# AUDIT LOG FUNCTIONS
def log_action(db, username, action, detail=""):

    entry = AuditLog(
        username=username,
        action=action,
        detail=detail
    )

    db.add(entry)
    db.flush()

    return entry.id


def get_audit_logs(db):

    logs = db.query(AuditLog).order_by(
        AuditLog.id.desc()
    ).limit(1000).all()

    return [
        {
            "id": log.id,
            "username": log.username,
            "action": log.action,
            "detail": log.detail,
            "timestamp": str(log.timestamp or "")
        }
        for log in logs
    ]


if __name__ == "__main__":

    init_db()

    db = SessionLocal()

    try:
        print("Database connection successful.")
        print("Users:", len(get_all_users(db)))
        print("Students:", len(get_students(db)))
        print("Audit logs:", len(get_audit_logs(db)))

    finally:
        db.close()
