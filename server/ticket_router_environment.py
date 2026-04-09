"""
Exoplanet Survey Environment — AstroEnvironment

A multi-step reinforcement learning environment where an AI agent acts as an
astrophysicist investigating candidate star systems for exoplanets.

The agent must strategically allocate limited telescope time to gather
observational data (transit photometry, radial velocity, spectroscopy)
before making a final planet classification.

Grading rewards both ACCURACY and EFFICIENCY:
  - Correct classification with thorough evidence → highest reward (~0.90)
  - Correct classification with minimal evidence  → good reward (~0.72)
  - Wrong classification                          → penalty (0.05)
  - Timeout without classifying                   → penalty (0.05)
"""

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
    Environment = object

# ──────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────

MAX_STEPS_PER_TASK = 5
CLASSIFICATIONS = ["Gas Giant", "Super Earth", "Terrestrial", "No Planet"]

# ──────────────────────────────────────────────────────────────
#  Simulated Star Systems (inspired by real Kepler/TESS targets)
# ──────────────────────────────────────────────────────────────

STAR_SYSTEMS = [
    {
        # Star 0 — Easy: Obvious Hot Jupiter (inspired by HD 209458 b)
        "id": 0,
        "name": "KOI-7921",
        "star_type": "G2V (Sun-like)",
        "star_mass_solar": 1.02,
        "star_temperature_k": 5780,
        "star_distance_ly": 870.0,
        "transit": {
            "depth_ppm": 14500,       # Very deep dip → big planet
            "duration_hours": 2.8,
            "period_days": 3.52,
        },
        "radial_velocity": {
            "amplitude_ms": 85.0,     # High → massive planet
            "min_mass_earth": 318.0,  # ~1 Jupiter mass
            "eccentricity": 0.02,
        },
        "spectroscopy": {
            "atmosphere_detected": True,
            "composition": "H2/He dominant, Na I D-line absorption, K features, no H2O",
            "surface_temp_k": 1400,
        },
        "target": "Gas Giant",
    },
    {
        # Star 1 — Medium: Super Earth in habitable zone (inspired by Kepler-442b)
        "id": 1,
        "name": "TOI-4830",
        "star_type": "K1V (Orange Dwarf)",
        "star_mass_solar": 0.78,
        "star_temperature_k": 5100,
        "star_distance_ly": 1206.0,
        "transit": {
            "depth_ppm": 830,         # Medium transit depth
            "duration_hours": 4.1,
            "period_days": 28.5,
        },
        "radial_velocity": {
            "amplitude_ms": 2.8,
            "min_mass_earth": 6.2,    # Several Earth masses
            "eccentricity": 0.04,
        },
        "spectroscopy": {
            "atmosphere_detected": True,
            "composition": "N2, CO2 traces, possible H2O vapor",
            "surface_temp_k": 295,    # Near Earth temperature!
        },
        "target": "Super Earth",
    },
    {
        # Star 2 — Medium: Terrestrial around red dwarf (inspired by TRAPPIST-1e)
        "id": 2,
        "name": "TIC-29140",
        "star_type": "M4V (Red Dwarf)",
        "star_mass_solar": 0.12,
        "star_temperature_k": 3050,
        "star_distance_ly": 40.7,
        "transit": {
            "depth_ppm": 380,         # Small but detectable
            "duration_hours": 0.9,
            "period_days": 6.1,
        },
        "radial_velocity": {
            "amplitude_ms": 1.1,
            "min_mass_earth": 1.4,    # Near Earth mass
            "eccentricity": 0.01,
        },
        "spectroscopy": {
            "atmosphere_detected": True,
            "composition": "Thin CO2 atmosphere, no H2O detected",
            "surface_temp_k": 220,
        },
        "target": "Terrestrial",
    },
    {
        # Star 3 — Hard: False positive / No planet (inspired by eclipsing binary)
        "id": 3,
        "name": "KIC-8462852",
        "star_type": "G5IV (Subgiant)",
        "star_mass_solar": 1.20,
        "star_temperature_k": 5490,
        "star_distance_ly": 1480.0,
        "transit": {
            "depth_ppm": 0,           # No transit signal at all
            "duration_hours": 0.0,
            "period_days": 0.0,
        },
        "radial_velocity": {
            "amplitude_ms": 0.3,      # Stellar noise only
            "min_mass_earth": 0.0,
            "eccentricity": 0.0,
        },
        "spectroscopy": {
            "atmosphere_detected": False,
            "composition": "N/A — stellar spectrum only, no planetary signatures",
            "surface_temp_k": 0,
        },
        "target": "No Planet",
    },
    {
        # Star 4 — Hard: Gas Giant with JWST–quality data (inspired by WASP-96b)
        "id": 4,
        "name": "WASP-96",
        "star_type": "G8V (Sun-like, metal-rich)",
        "star_mass_solar": 1.06,
        "star_temperature_k": 5500,
        "star_distance_ly": 1120.0,
        "transit": {
            "depth_ppm": 12200,       # Deep transit
            "duration_hours": 2.4,
            "period_days": 3.43,
        },
        "radial_velocity": {
            "amplitude_ms": 140.0,
            "min_mass_earth": 480.0,  # ~1.5 Jupiter masses
            "eccentricity": 0.06,
        },
        "spectroscopy": {
            "atmosphere_detected": True,
            "composition": "Clear H2/He atmosphere, strong H2O absorption at 1.4μm, Na, K — JWST benchmark",
            "surface_temp_k": 1285,
        },
        "target": "Gas Giant",
    },
]


# ──────────────────────────────────────────────────────────────
#  Environment
# ──────────────────────────────────────────────────────────────

class AstroEnvironment(Environment):
    """Multi-step exoplanet classification environment.

    The agent investigates one star system per episode.  At each step it
    can request an additional data product (transit, radial-velocity, or
    spectroscopy) or commit to a final classification.  The environment
    grades both accuracy AND the quality of evidence gathered.
    """
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, **kwargs):
        if Environment is not object:
            super().__init__(**kwargs)
        self._state = State(current_star_id=0, steps_used=0,
                            transit_revealed=False, rv_revealed=False,
                            spectro_revealed=False, classified=False,
                            total_correct=0)

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
        if episode_id is not None:
            try:
                star_id = int(str(episode_id).split("_")[-1]) % len(STAR_SYSTEMS)
            except ValueError:
                pass
        self._state = State(
            current_star_id=star_id,
            steps_used=0,
            transit_revealed=False,
            rv_revealed=False,
            spectro_revealed=False,
            classified=False,
            total_correct=0,
        )
        return self._get_observation()

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        self._state.steps_used += 1
        star = STAR_SYSTEMS[self._state.current_star_id]
        done = False

        # ── Handle each action type ──
        if action.action_type == "request_transit":
            if not self._state.transit_revealed:
                self._state.transit_revealed = True
                reward = 0.15  # Useful investigation step
            else:
                reward = 0.05  # Wasted step re-requesting data

        elif action.action_type == "request_radial_velocity":
            if not self._state.rv_revealed:
                self._state.rv_revealed = True
                reward = 0.15
            else:
                reward = 0.05

        elif action.action_type == "request_spectroscopy":
            if not self._state.spectro_revealed:
                self._state.spectro_revealed = True
                reward = 0.15
            else:
                reward = 0.05

        elif action.action_type == "classify":
            done = True
            self._state.classified = True
            if action.classification == star["target"]:
                reward = self._compute_accuracy_reward()
                self._state.total_correct += 1
            else:
                reward = 0.05  # Wrong classification

        else:
            reward = 0.05  # Invalid / unrecognised action

        # ── Timeout: used all steps without classifying ──
        if self._state.steps_used >= MAX_STEPS_PER_TASK and not done:
            done = True
            reward = 0.05

        obs = self._get_observation()
        obs.reward = reward
        obs.done = done
        return obs

    # ── Private helpers ──────────────────────────────────────

    def _compute_accuracy_reward(self) -> float:
        """Compute a nuanced reward that values both correctness and
        thoroughness of evidence, while also rewarding efficiency.

        Formula:
            score = base + data_bonus + efficiency_bonus
            clamped to [0.05, 0.95]
        """
        data_gathered = sum([
            self._state.transit_revealed,
            self._state.rv_revealed,
            self._state.spectro_revealed,
        ])
        base = 0.60
        data_bonus = data_gathered * 0.08        # up to +0.24
        steps_left = MAX_STEPS_PER_TASK - self._state.steps_used
        efficiency_bonus = max(0, steps_left) * 0.02  # up to +0.08
        return min(base + data_bonus + efficiency_bonus, 0.95)

    def _get_observation(self) -> Observation:
        star = STAR_SYSTEMS[self._state.current_star_id]

        # Build available-actions list (only show un-requested data)
        available_actions = []
        if not self._state.transit_revealed:
            available_actions.append("request_transit")
        if not self._state.rv_revealed:
            available_actions.append("request_radial_velocity")
        if not self._state.spectro_revealed:
            available_actions.append("request_spectroscopy")
        available_actions.append("classify")

        obs = Observation(
            star_name=star["name"],
            star_type=star["star_type"],
            star_mass_solar=star["star_mass_solar"],
            star_temperature_k=star["star_temperature_k"],
            star_distance_ly=star["star_distance_ly"],
            available_actions=available_actions,
            available_classifications=CLASSIFICATIONS,
            steps_remaining=MAX_STEPS_PER_TASK - self._state.steps_used,
            steps_used=self._state.steps_used,
            mission_phase="complete" if self._state.classified else "investigation",
        )

        # Reveal data that the agent has requested
        if self._state.transit_revealed:
            obs.transit_observed = True
            obs.transit_depth_ppm = star["transit"]["depth_ppm"]
            obs.transit_duration_hours = star["transit"]["duration_hours"]
            obs.orbital_period_days = star["transit"]["period_days"]

        if self._state.rv_revealed:
            obs.rv_observed = True
            obs.rv_amplitude_ms = star["radial_velocity"]["amplitude_ms"]
            obs.planet_min_mass_earth = star["radial_velocity"]["min_mass_earth"]
            obs.eccentricity = star["radial_velocity"]["eccentricity"]

        if self._state.spectro_revealed:
            obs.spectroscopy_observed = True
            obs.atmosphere_detected = star["spectroscopy"]["atmosphere_detected"]
            obs.atmosphere_composition = star["spectroscopy"]["composition"]
            obs.estimated_surface_temp_k = star["spectroscopy"]["surface_temp_k"]

        return obs