import logging
from datetime import datetime
from typing import Any, Dict, Optional
from sqlalchemy.orm import Session

from receptionist.database import SessionLocal
from receptionist.models import Employee, Meeting
from models.ollama_processor import OllamaProcessor

logger = logging.getLogger(__name__)


def _clear_meeting_state() -> None:
    meeting_state["visitor_id"] = None
    meeting_state["visitor_name"] = None
    meeting_state["employee_name"] = None


# State to track the visitor across turns
meeting_state: Dict[str, Any] = {
    "visitor_id": None,  # Optional[int]
    "visitor_name": None,
    "employee_name": None,
}


def create_initial_visitor(v_name: str) -> Optional[int]:
    """
    Creates a record in the 'visitors' table immediately.
    Returns the new row ID.
    """
    session: Session = SessionLocal()
    try:
        # Create entry with just the name and current time
        new_visit = Meeting(
            visitor_name=v_name,
            meeting_time=datetime.now(),
            status="Arrived",
        )
        session.add(new_visit)
        session.commit()
        session.refresh(new_visit)
        logger.info(f"INITIAL LOG: Visitor {v_name} registered with ID {new_visit.id}")
        return new_visit.id
    except Exception as e:
        session.rollback()
        logger.error(f"Failed initial log: {e}")
        return None
    finally:
        session.close()


def update_visitor_status(visitor_id: int, status: str) -> bool:
    """Updates status for an existing visitor record."""
    session: Session = SessionLocal()
    try:
        visit_record = session.query(Meeting).filter(Meeting.id == visitor_id).first()
        if not visit_record:
            return False
        visit_record.status = status
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        logger.error(f"Failed status update: {e}")
        return False
    finally:
        session.close()


def update_visitor_meeting(visitor_id: int, e_name: str) -> Optional[Dict]:
    """
    Updates the existing visitor record with the employee they want to meet.
    """
    session: Session = SessionLocal()
    try:
        # 1. Find the employee
        emp = session.query(Employee).filter(Employee.name.ilike(f"%{e_name}%")).first()
        if not emp:
            return {"error": "employee_not_found"}

        # 2. Find the visitor record we created earlier
        visit_record = session.query(Meeting).filter(Meeting.id == visitor_id).first()
        if visit_record:
            visit_record.employee_name = emp.name
            visit_record.status = "Meeting Scheduled"
            session.commit()

            return {"employee": emp.name, "cabin": emp.cabin_number}
        return None
    except Exception as e:
        session.rollback()
        return {"error": "db_error"}
    finally:
        session.close()


def _merge_entities(entities: Dict[str, Any], raw_query: str) -> None:
    """Smartly maps names based on context."""
    raw_query_lower = raw_query.lower()
    if entities.get("visitor_name") and not meeting_state.get("visitor_name"):
        meeting_state["visitor_name"] = entities["visitor_name"]
    if entities.get("employee_name") and not meeting_state.get("employee_name"):
        meeting_state["employee_name"] = entities["employee_name"]

    if entities.get("name"):
        val = entities["name"]
        if any(k in raw_query_lower for k in ["i am", "i'm", "myself", "this is"]):
            meeting_state["visitor_name"] = val
        elif any(
            k in raw_query_lower
            for k in ["meet", "meeting", "see", "here to see", "looking for"]
        ):
            meeting_state["employee_name"] = val
        else:
            if not meeting_state["visitor_name"]:
                meeting_state["visitor_name"] = val
            elif not meeting_state["employee_name"]:
                meeting_state["employee_name"] = val


async def route_query(user_query: str) -> str:
    ollama = OllamaProcessor.get_instance()
    extracted = await ollama.extract_intent_and_entities(user_query)
    entities = extracted.get("entities") or {}

    _merge_entities(entities, user_query)

    user_query_lower = user_query.lower()

    # --- STEP 1: REGISTER VISITOR IMMEDIATELY ---
    if meeting_state["visitor_name"] and meeting_state["visitor_id"] is None:
        # This person just introduced themselves. Log them NOW.
        v_id = create_initial_visitor(meeting_state["visitor_name"])
        meeting_state["visitor_id"] = v_id

        # If they haven't said who they are meeting yet, ask.
        if not meeting_state["employee_name"]:
            return f"Welcome {meeting_state['visitor_name']}. I've checked you in. Are you here to meet someone or for a delivery?"

    # --- STEP 2: UPDATE WITH MEETING INFO ---
    if meeting_state["visitor_id"] and meeting_state["employee_name"]:
        result = update_visitor_meeting(
            meeting_state["visitor_id"], meeting_state["employee_name"]
        )

        if result and "error" not in result:
            # Success! Clear state for next person
            v_name = meeting_state["visitor_name"]
            _clear_meeting_state()

            return f"Thank you {v_name}. I've updated your record. You can find {result['employee']} in cabin {result['cabin']}."

        if result and result.get("error") == "employee_not_found":
            attempted = meeting_state["employee_name"]
            meeting_state["employee_name"] = None  # Reset so they can try again
            return f"I've logged your arrival, but I couldn't find '{attempted}' in our directory. Who would you like to meet?"

    # --- STEP 3: HANDLING NON-MEETING VISITORS (Delivery/Interns) ---
    if meeting_state["visitor_id"] and not meeting_state["employee_name"]:
        # If they say "I'm just here for a delivery"
        if any(k in user_query_lower for k in ["delivery", "courier"]):
            v_name = meeting_state["visitor_name"]
            if update_visitor_status(meeting_state["visitor_id"], "Delivery Logged"):
                _clear_meeting_state()
                return f"Understood, {v_name}. I've logged your delivery. You can leave the package at the desk."
            return "I couldn't log that delivery right now—can you try again?"

        if "intern" in user_query_lower:
            v_name = meeting_state["visitor_name"]
            if update_visitor_status(meeting_state["visitor_id"], "Intern Visit"):
                _clear_meeting_state()
                return f"Thanks, {v_name}. I've logged your visit as an intern."
            return "I couldn't log that intern visit right now—can you try again?"

    return "Welcome! How can I help you today?"
