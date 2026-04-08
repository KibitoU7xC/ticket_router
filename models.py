from typing import List

from openenv.core.env_server.types import (
    Action as BaseAction,
    Observation as BaseObservation,
    State as BaseState,
)


class Action(BaseAction):
    # The agent must choose the correct department based on the ticket
    department: str


class Observation(BaseObservation):
    # What the agent sees: the complaint and the available options
    ticket_text: str = ""
    available_departments: List[str] = []


class State(BaseState):
    # Background tracking for the environment
    current_ticket_id: int = 0
    total_resolved: int = 0
    failed_attempts: int = 0