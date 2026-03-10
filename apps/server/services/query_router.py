import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from models.ollama_processor import OllamaProcessor
from receptionist.database import SessionLocal
from receptionist.models import Employee, Meeting

logger = logging.getLogger(__name__)

# Added role_lookup to the set of DB intents
DB_INTENTS = {"employee_lookup", "department_lookup", "cabin_lookup", "role_lookup"}
SCHEDULE_INTENT = "schedule_meeting"

# Accumulates meeting info across turns. Cleared after successful schedule.
meeting_state: Dict[str, Optional[str]] = {
    "visitor_name": None,
    "employee_name": None,
    "time": None,
}


def _serialize_employee(emp: Employee) -> Dict[str, Any]:
    return {
        "id": emp.id,
        "name": emp.name,
        "role": emp.role,
        "department": emp.department,
        "cabin_number": emp.cabin_number,
    }


def _parse_meeting_time(time_str: str) -> Optional[datetime]:
    """Parse meeting time from LLM-extracted string. Returns datetime or None."""
    if not time_str or not str(time_str).strip():
        return None
    s = str(time_str).strip()
    try:
        # 24h: "16:00", "16:00:00", "9:30"
        m = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$", s)
        if m:
            h, minute = int(m.group(1)), int(m.group(2))
            if 0 <= h <= 23 and 0 <= minute <= 59:
                base = datetime.now().replace(
                    hour=h, minute=minute, second=0, microsecond=0
                )
                return base
        # 12h: "4 PM", "4:30 PM", "9 AM"
        m = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*(AM|PM)$", s, re.IGNORECASE)
        if m:
            h, minute = int(m.group(1)), int(m.group(2) or 0)
            if m.group(3).upper() == "PM" and h != 12:
                h += 12
            elif m.group(3).upper() == "AM" and h == 12:
                h = 0
            if 0 <= h <= 23 and 0 <= minute <= 59:
                return datetime.now().replace(
                    hour=h, minute=minute, second=0, microsecond=0
                )
    except (ValueError, TypeError):
        pass
    return None


def _merge_schedule_entities(entities: Dict[str, Any]) -> None:
    """
    Merge extracted entities into meeting_state.
    Only updates fields that are explicitly provided (non-null, non-empty).
    """
    mapping = [
        ("visitor_name", "visitor_name"),
        ("employee_name", "employee_name"),
        ("time", "time"),
        ("meeting_time", "time"),
    ]
    for src_key, state_key in mapping:
        val = entities.get(src_key)
        if val is not None and str(val).strip():
            meeting_state[state_key] = str(val).strip()


def _get_missing_schedule_field() -> Optional[str]:
    """Returns the first missing field name, or None if all present."""
    if not meeting_state.get("visitor_name"):
        return "visitor_name"
    if not meeting_state.get("employee_name"):
        return "employee_name"
    if not meeting_state.get("time"):
        return "time"
    return None


def _ask_for_missing_field(field: str) -> str:
    """Conversational prompts when a required field is missing."""
    prompts = {
        "visitor_name": "May I have your name, please?",
        "employee_name": "Who would you like to meet?",
        "time": "What time would you like to schedule the meeting?",
    }
    return prompts.get(field, "I need a bit more information to schedule your meeting.")


def _create_meeting(
    visitor_name: str, employee_name: str, meeting_time: datetime, session: Session
) -> bool:
    """Insert a Meeting record. Returns True on success."""
    meeting = Meeting(
        visitor_name=visitor_name,
        employee_name=employee_name,
        meeting_time=meeting_time,
    )
    session.add(meeting)
    session.commit()
    return True


def handle_schedule_meeting() -> Optional[Dict[str, Any]]:
    """
    Schedule a meeting using meeting_state.
    """
    visitor_name = meeting_state.get("visitor_name") or ""
    employee_name = meeting_state.get("employee_name") or ""
    time_raw = meeting_state.get("time") or ""

    if not visitor_name.strip() or not employee_name.strip() or not time_raw.strip():
        return None

    meeting_dt = _parse_meeting_time(time_raw)
    if not meeting_dt:
        return None

    session: Session = SessionLocal()
    try:
        employee = (
            session.query(Employee)
            .filter(Employee.name.ilike(f"%{employee_name}%"))
            .first()
        )
        if not employee:
            return {"intent": SCHEDULE_INTENT, "error": "employee_not_found"}

        _create_meeting(visitor_name, employee.name, meeting_dt, session)
        result = {
            "intent": SCHEDULE_INTENT,
            "visitor_name": visitor_name,
            "employee": _serialize_employee(employee),
            "meeting_time": meeting_dt.isoformat(),
        }
        meeting_state["visitor_name"] = None
        meeting_state["employee_name"] = None
        meeting_state["time"] = None
        return result
    except Exception as e:
        logger.error(f"schedule_meeting failed: {e}")
        session.rollback()
        return {"intent": SCHEDULE_INTENT, "error": "database_error"}
    finally:
        session.close()


def handle_db_query(extracted_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    intent = extracted_data.get("intent", "general_conversation")
    entities = extracted_data.get("entities") or {}

    if intent == SCHEDULE_INTENT:
        return None

    session: Session = SessionLocal()
    try:
        # Get values from LLM extraction
        name_val = entities.get("name")
        role_val = entities.get("role")
        dept_val = entities.get("department")
        cabin_val = entities.get("cabin_number")

        # 1. SEARCH BY NAME OR ROLE (Flexible Search)
        # If the user says "Who is the HR Executive", the LLM might put it in 'name' OR 'role'
        search_term = name_val or role_val
        if search_term:
            # First, try searching the NAME column
            emp = (
                session.query(Employee)
                .filter(Employee.name.ilike(f"%{search_term}%"))
                .first()
            )

            # If not found by name, try searching the ROLE column
            if not emp:
                emp = (
                    session.query(Employee)
                    .filter(Employee.role.ilike(f"%{search_term}%"))
                    .first()
                )

            if emp:
                return {
                    "intent": "employee_lookup",
                    "employee": _serialize_employee(emp),
                }

        # 2. SEARCH BY DEPARTMENT
        if dept_val:
            employees = (
                session.query(Employee)
                .filter(Employee.department.ilike(f"%{dept_val}%"))
                .all()
            )
            if employees:
                return {
                    "intent": "department_lookup",
                    "department": dept_val,
                    "employees": [_serialize_employee(e) for e in employees],
                }

        # 3. SEARCH BY CABIN
        if cabin_val:
            emp = (
                session.query(Employee)
                .filter(Employee.cabin_number.ilike(f"%{cabin_val}%"))
                .first()
            )
            if emp:
                return {
                    "intent": "cabin_lookup",
                    "cabin_number": cabin_val,
                    "employee": _serialize_employee(emp),
                }

        return None
    finally:
        session.close()


def _format_schedule_success(result: Dict[str, Any]) -> str:
    """Build natural receptionist response for a successfully scheduled meeting."""
    visitor = result.get("visitor_name", "there")
    employee = result.get("employee", {})
    emp_name = employee.get("name", "")
    cabin = employee.get("cabin_number", "")
    cabin_part = f" {emp_name}'s cabin is {cabin}." if cabin else "."
    return f"Thank you {visitor}. Your meeting with {emp_name} has been scheduled.{cabin_part}"


async def route_query(user_query: str) -> str:
    """
    Main router: detect intent, dispatch to handler, return response.
    """
    if not user_query or not user_query.strip():
        return "How can I help you today?"

    ollama = OllamaProcessor.get_instance()
    extracted_data = await ollama.extract_intent_and_entities(user_query)
    intent = extracted_data.get("intent", "general_conversation")

    # 1. Schedule meeting intent
    if intent == SCHEDULE_INTENT:
        entities = extracted_data.get("entities") or {}
        _merge_schedule_entities(entities)

        missing = _get_missing_schedule_field()
        if missing:
            return _ask_for_missing_field(missing)

        schedule_result = handle_schedule_meeting()
        if schedule_result:
            if schedule_result.get("error") == "employee_not_found":
                return "Sorry, I couldn't find that employee in our directory."
            if "error" not in schedule_result:
                return _format_schedule_success(schedule_result)
            if schedule_result.get("error") == "database_error":
                return "I'm sorry, I had trouble saving your meeting. Please try again."

        if meeting_state.get("time"):
            return "I couldn't understand that time. What time would you like to schedule? (e.g. 4 PM or 16:00)"

    # 2. DB lookup intents (employee, role, department, cabin)
    db_result = handle_db_query(extracted_data)
    if db_result:
        return await ollama.generate_grounded_response(
            context=db_result,
            question=user_query,
        )

    # 3. General conversation fallback
    if intent == "general_conversation":
        return await ollama.get_response(user_query)

    return "I'm sorry, I couldn't find any matching records in our office database."
