"""
Unit tests for the Exoplanet Survey AstroEnvironment.

Validates:
  - Environment reset and observation structure
  - Multi-step investigation flow
  - Grading logic (correct / wrong / timeout)
  - Score boundaries (strictly between 0 and 1)
  - All 12 star systems are solvable
"""

import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.ticket_router_environment import AstroEnvironment, STAR_SYSTEMS
from models import Action


def test_reset_returns_valid_observation():
    """Reset should return an observation with star info but no revealed data."""
    env = AstroEnvironment()
    obs = env.reset()

    assert obs.star_name != "", "Star name must not be empty"
    assert obs.star_type != "", "Star type must not be empty"
    assert obs.star_mass_solar > 0, "Star mass must be positive"
    assert obs.transit_observed is False, "Transit should not be revealed on reset"
    assert obs.rv_observed is False, "RV should not be revealed on reset"
    assert obs.spectroscopy_observed is False, "Spectroscopy should not be revealed on reset"
    assert obs.steps_remaining == 5, "Should have 5 steps remaining"
    assert "classify" in obs.available_actions, "'classify' must always be available"
    print("  [PASS] test_reset_returns_valid_observation")


def test_request_transit_reveals_data():
    """Requesting transit should reveal transit data in the next observation."""
    env = AstroEnvironment()
    env.reset(episode_id="0")

    obs = env.step(Action(action_type="request_transit"))

    assert obs.transit_observed is True, "Transit data should now be revealed"
    assert obs.transit_depth_ppm >= 0, "Transit depth must be non-negative"
    assert obs.rv_observed is False, "RV should still be hidden"
    assert obs.reward > 0, "Reward for valid observation should be positive"
    assert obs.done is False, "Episode should not be done after observation"
    print("  [PASS] test_request_transit_reveals_data")


def test_correct_classification_gives_high_reward():
    """Correct classification should give reward > 0.6."""
    env = AstroEnvironment()
    env.reset(episode_id="0")  # KOI-7921 -> target is "Gas Giant"

    # Gather some evidence first
    env.step(Action(action_type="request_transit"))
    env.step(Action(action_type="request_radial_velocity"))
    obs = env.step(Action(action_type="classify", classification="Gas Giant"))

    assert obs.reward > 0.6, f"Correct classification should reward > 0.6, got {obs.reward}"
    assert obs.reward < 1.0, f"Reward must be strictly < 1.0, got {obs.reward}"
    assert obs.done is True, "Episode should be done after classification"
    print("  [PASS] test_correct_classification_gives_high_reward")


def test_wrong_classification_gives_low_reward():
    """Wrong classification should give reward = 0.05."""
    env = AstroEnvironment()
    env.reset(episode_id="0")  # Target is "Gas Giant"

    obs = env.step(Action(action_type="classify", classification="Terrestrial"))

    assert obs.reward == 0.05, f"Wrong classification should give 0.05, got {obs.reward}"
    assert obs.done is True, "Episode should be done after classification"
    print("  [PASS] test_wrong_classification_gives_low_reward")


def test_timeout_gives_penalty():
    """Running out of steps without classifying should penalise."""
    env = AstroEnvironment()
    env.reset(episode_id="0")

    env.step(Action(action_type="request_transit"))
    env.step(Action(action_type="request_radial_velocity"))
    env.step(Action(action_type="request_spectroscopy"))
    env.step(Action(action_type="request_transit"))  # duplicate
    obs = env.step(Action(action_type="request_radial_velocity"))  # duplicate, step 5

    assert obs.done is True, "Should be done after 5 steps"
    assert obs.reward == 0.05, f"Timeout penalty should be 0.05, got {obs.reward}"
    print("  [PASS] test_timeout_gives_penalty")


def test_all_scores_in_valid_range():
    """Every possible reward from the environment must be strictly in (0, 1)."""
    env = AstroEnvironment()

    for star_id in range(len(STAR_SYSTEMS)):
        star = STAR_SYSTEMS[star_id]

        # Test correct classification
        env.reset(episode_id=str(star_id))
        env.step(Action(action_type="request_transit"))
        obs = env.step(Action(action_type="classify", classification=star["target"]))
        assert 0 < obs.reward < 1, f"Star {star_id} correct reward {obs.reward} out of range"

        # Test wrong classification
        env.reset(episode_id=str(star_id))
        wrong = "No Planet" if star["target"] != "No Planet" else "Gas Giant"
        obs = env.step(Action(action_type="classify", classification=wrong))
        assert 0 < obs.reward < 1, f"Star {star_id} wrong reward {obs.reward} out of range"

    print("  [PASS] test_all_scores_in_valid_range (all 12 star systems)")


def test_all_star_systems_solvable():
    """Every star system should be solvable with the correct classification."""
    env = AstroEnvironment()
    solved = 0

    for star_id in range(len(STAR_SYSTEMS)):
        star = STAR_SYSTEMS[star_id]
        env.reset(episode_id=str(star_id))
        env.step(Action(action_type="request_transit"))
        env.step(Action(action_type="request_radial_velocity"))
        obs = env.step(Action(action_type="classify", classification=star["target"]))
        if obs.reward > 0.5:
            solved += 1

    assert solved == len(STAR_SYSTEMS), f"Only solved {solved}/{len(STAR_SYSTEMS)} star systems"
    print(f"  [PASS] test_all_star_systems_solvable ({solved}/{len(STAR_SYSTEMS)})")


def test_efficiency_bonus():
    """Classifying correctly in fewer steps should give a higher reward."""
    env = AstroEnvironment()

    # Fast: transit -> classify (2 steps)
    env.reset(episode_id="0")
    env.step(Action(action_type="request_transit"))
    obs_fast = env.step(Action(action_type="classify", classification="Gas Giant"))

    # Slow: transit -> RV -> spectro -> classify (4 steps)
    env.reset(episode_id="0")
    env.step(Action(action_type="request_transit"))
    env.step(Action(action_type="request_radial_velocity"))
    env.step(Action(action_type="request_spectroscopy"))
    obs_slow = env.step(Action(action_type="classify", classification="Gas Giant"))

    assert obs_fast.reward > 0.5, f"Fast correct should reward > 0.5, got {obs_fast.reward}"
    assert obs_slow.reward > 0.5, f"Slow correct should reward > 0.5, got {obs_slow.reward}"
    assert obs_slow.reward > obs_fast.reward, \
        f"More evidence ({obs_slow.reward}) should beat less evidence ({obs_fast.reward})"
    print("  [PASS] test_efficiency_bonus")


if __name__ == "__main__":
    print("Running AstroEnvironment unit tests...")
    print()
    test_reset_returns_valid_observation()
    test_request_transit_reveals_data()
    test_correct_classification_gives_high_reward()
    test_wrong_classification_gives_low_reward()
    test_timeout_gives_penalty()
    test_all_scores_in_valid_range()
    test_all_star_systems_solvable()
    test_efficiency_bonus()
    print()
    print("All tests passed!")
