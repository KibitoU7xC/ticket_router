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

# Our simulated Exoplanet dataset
MOCK_STARS = [
    {
        "id": 0, 
        "star_type": "G-type Main Sequence (Sun-like)",
        "transit_depth_percent": 1.25, 
        "orbital_period_days": 4.5,
        "star_mass_solar": 1.0,
        "target": "Gas Giant"  # High transit depth means large planet
    },
    {
        "id": 1, 
        "star_type": "M-type Dwarf (Red Dwarf)",
        "transit_depth_percent": 0.50, 
        "orbital_period_days": 12.0,
        "star_mass_solar": 0.3,
        "target": "Super Earth" # Medium depth
    },
    {
        "id": 2, 
        "star_type": "K-type Main Sequence",
        "transit_depth_percent": 0.05, 
        "orbital_period_days": 35.0,
        "star_mass_solar": 0.8,
        "target": "Terrestrial" # Tiny depth means Earth-like rock
    }
]
AVAILABLE_CLASSES = ["Gas Giant", "Super Earth", "Terrestrial", "No Planet"]


class AstroEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, **kwargs):
        if Environment is not object:
            super().__init__(**kwargs)
        self._state = State(current_star_id=0, total_resolved=0, failed_attempts=0)

    @property
    def state(self) -> State:
        return self._state

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        star_id = 0
        if episode_id:
            try:
                star_id = int(str(episode_id).split("_")[-1]) % len(MOCK_STARS)
            except ValueError:
                pass
        self._state = State(current_star_id=star_id, total_resolved=0, failed_attempts=0)
        return self._get_observation()

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        current_star = MOCK_STARS[self._state.current_star_id]
        
        # Grading Logic
        if action.planet_classification == current_star["target"]:
            reward = 0.95  # strictly between 0 and 1
            self._state.total_resolved += 1
            self._state.failed_attempts = 0
            # Move to next star
            self._state.current_star_id = (self._state.current_star_id + 1) % len(MOCK_STARS)
            done = self._state.total_resolved >= len(MOCK_STARS)
        else:
            reward = 0.05  # Penalty for wrong prediction strictly between 0 and 1
            self._state.failed_attempts += 1
            done = self._state.failed_attempts >= 3

        obs = self._get_observation()
        obs.reward = reward
        obs.done = done
        return obs

    def _get_observation(self) -> Observation:
        next_star = MOCK_STARS[self._state.current_star_id]
        return Observation(
            star_type=next_star["star_type"],
            transit_depth_percent=next_star["transit_depth_percent"],
            orbital_period_days=next_star["orbital_period_days"],
            star_mass_solar=next_star["star_mass_solar"],
            available_classifications=AVAILABLE_CLASSES,
        )