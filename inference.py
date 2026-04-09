"""
Exoplanet Survey — Inference Script

A ReAct-style (Reasoning + Acting) agent that conducts multi-step
astrophysical investigations using an LLM.  For each star system the agent
strategically requests telescope data before committing to a planet
classification.

Structured output format:
    [START] task=<name>
    [STEP]  step=<n> reward=<r>
    [END]   task=<name> score=<s> steps=<n>
"""

import os
import json
import re
import sys
from openai import OpenAI
from server.ticket_router_environment import AstroEnvironment
from models import Action

# ──────────────────────────────────────────────────────────────
#  Structured-output emitters  (Phase-2 validator compliance)
# ──────────────────────────────────────────────────────────────

def emit_start(task_name: str):
    print(f"[START] task={task_name}", flush=True)

def emit_step(step: int, reward: float):
    print(f"[STEP] step={step} reward={reward}", flush=True)

def emit_end(task_name: str, score: float, steps: int):
    print(f"[END] task={task_name} score={score} steps={steps}", flush=True)


# ──────────────────────────────────────────────────────────────
#  LLM Prompting
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert astrophysicist conducting an exoplanet survey mission.
You investigate one star system at a time using limited telescope time.

AT EACH STEP choose exactly ONE action:
  • "request_transit"          — Observe the light-curve for transiting planets
  • "request_radial_velocity"  — Measure Doppler shifts for planet mass
  • "request_spectroscopy"     — Analyse the atmosphere via spectral lines
  • "classify"                 — Commit your final planet classification

CLASSIFICATION OPTIONS: Gas Giant | Super Earth | Terrestrial | No Planet

DECISION RULES (based on real astrophysics):
  Transit depth > 10 000 ppm, mass > 100 M⊕          → Gas Giant
  Transit depth 200–5 000 ppm, mass 2–15 M⊕           → Super Earth
  Transit depth < 500 ppm, mass < 2 M⊕                → Terrestrial
  Transit depth ≈ 0 ppm AND negligible RV amplitude   → No Planet

STRATEGY:
  1. ALWAYS request transit data first — it is the most diagnostic.
  2. If transit depth is 0 or near zero → classify "No Planet" immediately.
  3. Otherwise request radial velocity to pin down the mass.
  4. Only request spectroscopy if the case is ambiguous or you have steps left.
  5. You have ≤ 5 steps total — budget wisely!

── Example investigation ──
Star: G2V, 1.0 M☉, 5780 K
Step 1 → {"action_type": "request_transit"}
         Result: depth = 15000 ppm, period = 3.5 d
Step 2 → {"action_type": "request_radial_velocity"}
         Result: amplitude = 100 m/s, min mass = 350 M⊕
Step 3 → {"action_type": "classify", "classification": "Gas Giant"}
Reasoning: Deep transit + high mass = Gas Giant.  ✓

Respond with ONLY a JSON object.  No explanation outside the JSON.
  Investigation: {"action_type": "request_transit"}
  Classification: {"action_type": "classify", "classification": "Gas Giant"}
"""


def build_observation_prompt(obs) -> str:
    """Format the current observation state into a human-readable prompt."""
    lines = [
        f"═══ Star System: {obs.star_name} ═══",
        f"Star Type      : {obs.star_type}",
        f"Star Mass       : {obs.star_mass_solar} M☉",
        f"Star Temperature: {obs.star_temperature_k} K",
        f"Distance        : {obs.star_distance_ly} ly",
        "",
    ]

    # Transit data
    if obs.transit_observed:
        lines += [
            "▸ TRANSIT DATA (observed):",
            f"    Transit Depth    : {obs.transit_depth_ppm} ppm",
            f"    Transit Duration : {obs.transit_duration_hours} hours",
            f"    Orbital Period   : {obs.orbital_period_days} days",
            "",
        ]
    else:
        lines += ["▸ Transit Data: NOT YET OBSERVED", ""]

    # Radial velocity
    if obs.rv_observed:
        lines += [
            "▸ RADIAL VELOCITY (observed):",
            f"    RV Amplitude     : {obs.rv_amplitude_ms} m/s",
            f"    Min Planet Mass   : {obs.planet_min_mass_earth} M⊕",
            f"    Eccentricity     : {obs.eccentricity}",
            "",
        ]
    else:
        lines += ["▸ Radial Velocity: NOT YET OBSERVED", ""]

    # Spectroscopy
    if obs.spectroscopy_observed:
        lines += [
            "▸ SPECTROSCOPY (observed):",
            f"    Atmosphere Detected : {obs.atmosphere_detected}",
            f"    Composition         : {obs.atmosphere_composition}",
            f"    Est. Surface Temp   : {obs.estimated_surface_temp_k} K",
            "",
        ]
    else:
        lines += ["▸ Spectroscopy: NOT YET OBSERVED", ""]

    lines += [
        f"Steps Remaining      : {obs.steps_remaining}",
        f"Available Actions    : {obs.available_actions}",
        f"Classification Options: {obs.available_classifications}",
        "",
        'Respond with ONLY JSON: {"action_type": "...", "classification": "..."}',
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
#  Robust LLM response parsing
# ──────────────────────────────────────────────────────────────

VALID_ACTIONS = {"request_transit", "request_radial_velocity", "request_spectroscopy", "classify"}
VALID_CLASSES = {"Gas Giant", "Super Earth", "Terrestrial", "No Planet"}


def parse_llm_response(raw: str) -> dict:
    """Extract action_type (and optional classification) from LLM output,
    with multi-layer fallbacks for noisy model responses."""

    # ── Layer 1: strict JSON ──
    try:
        data = json.loads(raw)
        at = data.get("action_type", "").strip()
        if at in VALID_ACTIONS:
            result = {"action_type": at}
            if at == "classify":
                cl = data.get("classification", "").strip()
                result["classification"] = cl if cl in VALID_CLASSES else _fuzzy_class(cl)
            return result
    except (json.JSONDecodeError, AttributeError):
        pass

    # ── Layer 2: regex extraction ──
    m_at = re.search(r'"action_type"\s*:\s*"([^"]+)"', raw)
    if m_at and m_at.group(1).strip() in VALID_ACTIONS:
        at = m_at.group(1).strip()
        result = {"action_type": at}
        if at == "classify":
            m_cl = re.search(r'"classification"\s*:\s*"([^"]+)"', raw)
            cl = m_cl.group(1).strip() if m_cl else ""
            result["classification"] = cl if cl in VALID_CLASSES else _fuzzy_class(cl)
        return result

    # ── Layer 3: keyword detection ──
    raw_lower = raw.lower()
    if "classify" in raw_lower:
        for vc in VALID_CLASSES:
            if vc.lower() in raw_lower:
                return {"action_type": "classify", "classification": vc}
        return {"action_type": "classify", "classification": "No Planet"}

    for action_key in ["request_spectroscopy", "request_radial_velocity", "request_transit"]:
        if action_key.replace("_", " ") in raw_lower or action_key in raw_lower:
            return {"action_type": action_key}

    # ── Layer 4: safe fallback ──
    return {"action_type": "request_transit"}


def _fuzzy_class(text: str) -> str:
    """Best-effort match of a string to a valid classification."""
    tl = text.lower()
    for vc in VALID_CLASSES:
        if vc.lower() in tl:
            return vc
    return "No Planet"


def call_llm(client: OpenAI, model: str, obs) -> dict:
    """Send the observation to the LLM and return parsed action dict."""
    user_prompt = build_observation_prompt(obs)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=100,
    )
    raw = response.choices[0].message.content.strip()
    return parse_llm_response(raw)


# ──────────────────────────────────────────────────────────────
#  Main loop — multi-task, multi-step ReAct agent
# ──────────────────────────────────────────────────────────────

import random

NUM_TASKS = 12  # One per star system


def main():
    hf_token = os.environ.get("HF_TOKEN")
    api_base_url = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1/")
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

    if not hf_token:
        print("ERROR: HF_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=api_base_url, api_key=hf_token)
    env = AstroEnvironment()

    # Randomised task order for variety (deterministic seed for reproducibility)
    task_order = list(range(NUM_TASKS))
    random.seed(42)
    random.shuffle(task_order)

    for task_idx in range(NUM_TASKS):
        star_id = task_order[task_idx]
        task_name = f"exoplanet_survey_{star_id}"
        obs = env.reset(episode_id=str(star_id))

        emit_start(task_name)

        step = 0
        final_score = 0.05

        # ── Multi-step investigation loop ──
        while obs.steps_remaining > 0 and obs.mission_phase != "complete":
            step += 1

            try:
                action_data = call_llm(client, model_name, obs)
            except Exception as e:
                print(f"LLM error at {task_name} step {step}: {e}", file=sys.stderr)
                emit_step(step, 0.05)
                final_score = 0.05
                break

            action = Action(**action_data)
            obs = env.step(action)

            emit_step(step, obs.reward)
            final_score = obs.reward

            if obs.done:
                break

        emit_end(task_name=task_name, score=round(final_score, 2), steps=step)


if __name__ == "__main__":
    main()