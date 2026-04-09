from typing import List

from openenv.core.env_server.types import (
    Action as BaseAction,
    Observation as BaseObservation,
    State as BaseState,
)

class Action(BaseAction):
    # The agent must choose the correct planet classification based on simulated data
    planet_classification: str


class Observation(BaseObservation):
    # Astrophysics data provided to the AI
    star_type: str = ""
    transit_depth_percent: float = 0.0
    orbital_period_days: float = 0.0
    star_mass_solar: float = 0.0
    available_classifications: List[str] = [
        "Gas Giant",
        "Super Earth",
        "Terrestrial",
        "No Planet"
    ]


class State(BaseState):
    # Background tracking for the environment
    current_star_id: int = 0
    total_resolved: int = 0
    failed_attempts: int = 0