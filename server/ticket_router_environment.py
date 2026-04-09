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
    {
        # Star 5 — Medium: Habitable-zone Super Earth (inspired by TOI-700d)
        "id": 5,
        "name": "TOI-700",
        "star_type": "M2V (Red Dwarf)",
        "star_mass_solar": 0.42,
        "star_temperature_k": 3480,
        "star_distance_ly": 101.4,
        "transit": {
            "depth_ppm": 680,
            "duration_hours": 3.2,
            "period_days": 37.4,
        },
        "radial_velocity": {
            "amplitude_ms": 1.6,
            "min_mass_earth": 3.6,
            "eccentricity": 0.06,
        },
        "spectroscopy": {
            "atmosphere_detected": True,
            "composition": "N2 dominant, CO2, possible O3 — Biosignature candidate",
            "surface_temp_k": 268,
        },
        "target": "Super Earth",
    },
    {
        # Star 6 — Hard: Neptune-mass but classified as Super Earth (inspired by HAT-P-11b)
        "id": 6,
        "name": "HAT-P-11",
        "star_type": "K4V (Orange Dwarf)",
        "star_mass_solar": 0.81,
        "star_temperature_k": 4780,
        "star_distance_ly": 123.0,
        "transit": {
            "depth_ppm": 4200,         # Borderline deep — tricky!
            "duration_hours": 2.3,
            "period_days": 4.89,
        },
        "radial_velocity": {
            "amplitude_ms": 11.6,
            "min_mass_earth": 12.0,    # 12 M⊕ — right at the Super Earth boundary
            "eccentricity": 0.26,      # Unusually eccentric
        },
        "spectroscopy": {
            "atmosphere_detected": True,
            "composition": "H2O detected via transmission, He outflow, no thick H2 envelope",
            "surface_temp_k": 870,
        },
        "target": "Super Earth",
    },
    {
        # Star 7 — Medium: Classic water-world Super Earth (inspired by GJ 1214b)
        "id": 7,
        "name": "GJ-1214",
        "star_type": "M4.5V (Red Dwarf)",
        "star_mass_solar": 0.15,
        "star_temperature_k": 3026,
        "star_distance_ly": 48.0,
        "transit": {
            "depth_ppm": 1430,
            "duration_hours": 0.87,
            "period_days": 1.58,
        },
        "radial_velocity": {
            "amplitude_ms": 12.2,
            "min_mass_earth": 8.2,
            "eccentricity": 0.0,
        },
        "spectroscopy": {
            "atmosphere_detected": True,
            "composition": "Thick hazy atmosphere, likely high mean-molecular-weight (H2O/steam)",
            "surface_temp_k": 550,
        },
        "target": "Super Earth",
    },
    {
        # Star 8 — Hard: Ultra-short-period rocky planet (inspired by Kepler-10b)
        "id": 8,
        "name": "Kepler-10",
        "star_type": "G2V (Sun-like)",
        "star_mass_solar": 0.91,
        "star_temperature_k": 5627,
        "star_distance_ly": 608.0,
        "transit": {
            "depth_ppm": 69,           # Extremely shallow — easy to miss
            "duration_hours": 1.81,
            "period_days": 0.84,       # Ultra-short period — hint at rocky
        },
        "radial_velocity": {
            "amplitude_ms": 3.3,
            "min_mass_earth": 1.9,     # Right at Earth/Super-Earth boundary
            "eccentricity": 0.0,
        },
        "spectroscopy": {
            "atmosphere_detected": False,
            "composition": "No significant atmosphere — likely bare rock (lava world)",
            "surface_temp_k": 2170,
        },
        "target": "Terrestrial",
    },
    {
        # Star 9 — Hard: False alarm with very low RV (inspired by Barnard's Star controversy)
        "id": 9,
        "name": "HD-164922",
        "star_type": "G9V (Sun-like)",
        "star_mass_solar": 0.87,
        "star_temperature_k": 5293,
        "star_distance_ly": 72.1,
        "transit": {
            "depth_ppm": 0,
            "duration_hours": 0.0,
            "period_days": 0.0,
        },
        "radial_velocity": {
            "amplitude_ms": 0.8,       # Very marginal — mostly stellar activity
            "min_mass_earth": 0.0,
            "eccentricity": 0.0,
        },
        "spectroscopy": {
            "atmosphere_detected": False,
            "composition": "No planetary signatures, pure stellar chromospheric activity",
            "surface_temp_k": 0,
        },
        "target": "No Planet",
    },
    {
        # Star 10 — Hard: Ultra-hot Jupiter around hot star (inspired by KELT-9b)
        "id": 10,
        "name": "KELT-9",
        "star_type": "B9.5V (Blue-white hot star)",
        "star_mass_solar": 2.52,
        "star_temperature_k": 10170,    # Very hot star — unusual host
        "star_distance_ly": 667.0,
        "transit": {
            "depth_ppm": 6780,          # Deep, but less than expected for Gas Giant
            "duration_hours": 3.9,      # because the star itself is huge
            "period_days": 1.48,
        },
        "radial_velocity": {
            "amplitude_ms": 275.0,      # Very high — massive planet
            "min_mass_earth": 920.0,    # ~2.9 Jupiter masses
            "eccentricity": 0.0,
        },
        "spectroscopy": {
            "atmosphere_detected": True,
            "composition": "Fe, Ti, Mg vaporised in atmosphere; T_day > 4000 K — hottest known planet",
            "surface_temp_k": 4050,
        },
        "target": "Gas Giant",
    },
    {
        # Star 11 — Hard: Habitable-zone Super Earth with possible biosignatures (inspired by LHS 1140b)
        "id": 11,
        "name": "LHS-1140",
        "star_type": "M4.5V (Red Dwarf)",
        "star_mass_solar": 0.18,
        "star_temperature_k": 3131,
        "star_distance_ly": 48.8,
        "transit": {
            "depth_ppm": 3460,         # Deep, but star is very small
            "duration_hours": 2.0,
            "period_days": 24.7,
        },
        "radial_velocity": {
            "amplitude_ms": 5.3,
            "min_mass_earth": 5.6,
            "eccentricity": 0.0,
        },
        "spectroscopy": {
            "atmosphere_detected": True,
            "composition": "Possible N2-dominated with CO2, hints of H2O — JWST priority target",
            "surface_temp_k": 230,
        },
        "target": "Super Earth",
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