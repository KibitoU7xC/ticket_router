# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ticket Router Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import TicketRouterAction, TicketRouterObservation


class TicketRouterEnv(
    EnvClient[TicketRouterAction, TicketRouterObservation, State]
):
    """
    Client for the Ticket Router Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with TicketRouterEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(TicketRouterAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = TicketRouterEnv.from_docker_image("ticket_router-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(TicketRouterAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: TicketRouterAction) -> Dict:
        """
        Convert TicketRouterAction to JSON payload for step message.

        Args:
            action: TicketRouterAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "planet_classification": action.planet_classification,
        }

    def _parse_result(self, payload: Dict) -> StepResult[TicketRouterObservation]:
        """
        Parse server response into StepResult[TicketRouterObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with TicketRouterObservation
        """
        obs_data = payload.get("observation", {})
        observation = TicketRouterObservation(
            star_type=obs_data.get("star_type", ""),
            transit_depth_percent=obs_data.get("transit_depth_percent", 0.0),
            orbital_period_days=obs_data.get("orbital_period_days", 0.0),
            star_mass_solar=obs_data.get("star_mass_solar", 0.0),
            available_classifications=obs_data.get("available_classifications", []),
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
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
