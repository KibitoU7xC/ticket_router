from typing import Any, Optional

try:
    from ..models import Action, Observation, State
except (ImportError, ValueError):
    try:
        from models import Action, Observation, State
    except ImportError:
        from ticket_router.models import Action, Observation, State

try:
    from openenv.core.env_server.interfaces import Environment
except ImportError:
    Environment = object  # Fallback for local testing without openenv

# Our mock dataset
MOCK_TICKETS = [
    {"id": 0, "text": "My screen is completely black, I cannot see anything.", "target": "Tech Support"},
    {"id": 1, "text": "I was overcharged $15 on my last invoice and want my money back.", "target": "Refunds"},
    {"id": 2, "text": "I can't log in because the app keeps crashing, so just cancel my entire subscription.", "target": "Tech Support"} 
]
AVAILABLE_DEPTS = ["Tech Support", "Refunds", "Billing", "General Inquiry"]


class TicketRouterEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, **kwargs):
        if Environment is not object:
            super().__init__(**kwargs)
        self._state = State(current_ticket_id=0, total_resolved=0, failed_attempts=0)

    @property
    def state(self) -> State:
        return self._state

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        self._state = State(current_ticket_id=0, total_resolved=0, failed_attempts=0)
        return self._get_observation()

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        current_ticket = MOCK_TICKETS[self._state.current_ticket_id]
        
        # Grading Logic
        if action.department == current_ticket["target"]:
            reward = 1.0
            self._state.total_resolved += 1
            self._state.failed_attempts = 0
            # Move to next ticket
            self._state.current_ticket_id = (self._state.current_ticket_id + 1) % len(MOCK_TICKETS)
            # Done only when all tickets have been resolved
            done = self._state.total_resolved >= len(MOCK_TICKETS)
        else:
            reward = -0.5  # Penalty for wrong department
            self._state.failed_attempts += 1
            # End episode if they fail too many times on the same ticket
            done = self._state.failed_attempts >= 3

        obs = self._get_observation()
        obs.reward = reward
        obs.done = done
        return obs

    def _get_observation(self) -> Observation:
        next_ticket = MOCK_TICKETS[self._state.current_ticket_id]
        return Observation(
            ticket_text=next_ticket["text"],
            available_departments=AVAILABLE_DEPTS,
        )