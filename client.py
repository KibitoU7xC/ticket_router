# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exoplanet Survey Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import Action as TicketRouterAction, Observation as TicketRouterObservation


class TicketRouterEnv(
    EnvClient[TicketRouterAction, TicketRouterObservation, State]
):
    """
    Client for the Exoplanet Survey Environment.

    This client maintains a persistent WebSocket connection to the environment
    server, enabling efficient multi-step investigations with low latency.

    Example:
        >>> with TicketRouterEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     # Request transit data
        ...     result = env.step(TicketRouterAction(action_type="request_transit"))
        ...     # Classify
        ...     result = env.step(TicketRouterAction(action_type="classify", classification="Gas Giant"))
    """

    def _step_payload(self, action: TicketRouterAction) -> Dict:
        """Convert Action to JSON payload for the step message."""
        payload = {"action_type": action.action_type}
        if action.action_type == "classify":
            payload["classification"] = action.classification
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[TicketRouterObservation]:
        """Parse server response into StepResult."""
        obs_data = payload.get("observation", {})
        observation = TicketRouterObservation(
            star_name=obs_data.get("star_name", ""),
            star_type=obs_data.get("star_type", ""),
            star_mass_solar=obs_data.get("star_mass_solar", 0.0),
            star_temperature_k=obs_data.get("star_temperature_k", 0),
            star_distance_ly=obs_data.get("star_distance_ly", 0.0),
            transit_observed=obs_data.get("transit_observed", False),
            transit_depth_ppm=obs_data.get("transit_depth_ppm", 0.0),
            transit_duration_hours=obs_data.get("transit_duration_hours", 0.0),
            orbital_period_days=obs_data.get("orbital_period_days", 0.0),
            rv_observed=obs_data.get("rv_observed", False),
            rv_amplitude_ms=obs_data.get("rv_amplitude_ms", 0.0),
            planet_min_mass_earth=obs_data.get("planet_min_mass_earth", 0.0),
            eccentricity=obs_data.get("eccentricity", 0.0),
            spectroscopy_observed=obs_data.get("spectroscopy_observed", False),
            atmosphere_detected=obs_data.get("atmosphere_detected", False),
            atmosphere_composition=obs_data.get("atmosphere_composition", ""),
            estimated_surface_temp_k=obs_data.get("estimated_surface_temp_k", 0),
            available_actions=obs_data.get("available_actions", []),
            available_classifications=obs_data.get("available_classifications", []),
            steps_remaining=obs_data.get("steps_remaining", 0),
            steps_used=obs_data.get("steps_used", 0),
            mission_phase=obs_data.get("mission_phase", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
