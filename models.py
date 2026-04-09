from typing import List

from openenv.core.env_server.types import (
    Action as BaseAction,
    Observation as BaseObservation,
    State as BaseState,
)


class Action(BaseAction):
    """Agent action for the Exoplanet Survey Mission.

    The agent can either request more observational data from the telescope
    or make a final planet classification.
    """
    action_type: str  # "request_transit", "request_radial_velocity", "request_spectroscopy", "classify"
    classification: str = ""  # Required when action_type == "classify"


class Observation(BaseObservation):
    """Telescope observation data from the Exoplanet Survey.

    Data fields are progressively revealed as the agent requests observations,
    simulating real telescope time allocation.
    """
    # Star identification (always visible)
    star_name: str = ""
    star_type: str = ""
    star_mass_solar: float = 0.0
    star_temperature_k: int = 0
    star_distance_ly: float = 0.0

    # Transit photometry (revealed on request)
    transit_observed: bool = False
    transit_depth_ppm: float = 0.0
    transit_duration_hours: float = 0.0
    orbital_period_days: float = 0.0

    # Radial velocity (revealed on request)
    rv_observed: bool = False
    rv_amplitude_ms: float = 0.0
    planet_min_mass_earth: float = 0.0
    eccentricity: float = 0.0

    # Spectroscopy (revealed on request)
    spectroscopy_observed: bool = False
    atmosphere_detected: bool = False
    atmosphere_composition: str = ""
    estimated_surface_temp_k: int = 0

    # Mission control metadata
    available_actions: List[str] = []
    available_classifications: List[str] = []
    steps_remaining: int = 5
    steps_used: int = 0
    mission_phase: str = "investigation"


class State(BaseState):
    """Internal state tracking for the Exoplanet Survey environment."""
    current_star_id: int = 0
    steps_used: int = 0
    transit_revealed: bool = False
    rv_revealed: bool = False
    spectro_revealed: bool = False
    classified: bool = False
    total_correct: int = 0