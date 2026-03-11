from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class Employee(Base):
    __tablename__ = "employees"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    role = Column(String)  # Add this: "Manager", "Head", "Intern", etc.
    department = Column(String)
    cabin_number = Column(String)


class Meeting(Base):
    __tablename__ = "visitors"

    id = Column(Integer, primary_key=True)
    visitor_name = Column("name", String)
    employee_name = Column("meeting_with", String)
    meeting_time = Column("checkin_time", DateTime, default=datetime.utcnow)
    status = Column(String, default="Arrived")
