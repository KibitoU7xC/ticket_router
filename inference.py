import os
import json
import re
import sys
from openai import OpenAI
from server.ticket_router_environment import AstroEnvironment
from models import Action

# ──────────────────────────────────────────────────────────────
#  Structured-output helpers (required by the Phase-2 validator)
# ──────────────────────────────────────────────────────────────

def emit_start(task_name: str):
    """Print the [START] block the validator looks for."""
    print(f"[START] task={task_name}", flush=True)


def emit_step(step: int, reward: float):
    """Print one [STEP] block per environment step."""
    print(f"[STEP] step={step} reward={reward}", flush=True)


def emit_end(task_name: str, score: float, steps: int):
    """Print the [END] block that closes the episode."""
    print(f"[END] task={task_name} score={score} steps={steps}", flush=True)


# ──────────────────────────────────────────────────────────────
#  LLM interaction
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert astrophysicist classifying exoplanets based on telescope data.
Your ONLY job is to classify the planet based on the star's transit depth and mass.

Rules for Classification:
- "Gas Giant": Huge dips in star brightness (transit_depth_percent > 1.0%)
- "Super Earth": Medium dips in brightness (transit_depth_percent between 0.1% and 1.0%)
- "Terrestrial": Tiny dips, like Earth (transit_depth_percent < 0.1%)
- "No Planet": No dip / transit_depth is exactly 0.0

Respond with ONLY a JSON object: {"planet_classification": "<chosen class>"}
Do NOT add any other text or reasoning.
"""


def build_user_prompt(obs) -> str:
    parts = [
        f'Star Type: {obs.star_type}',
        f'Transit Depth (%): {obs.transit_depth_percent}',
        f'Orbital Period (Days): {obs.orbital_period_days}',
        f'Star Mass (Solar): {obs.star_mass_solar}',
        f"Available Classes: {obs.available_classifications}",
        'Respond with ONLY: {"planet_classification": "..."}'
    ]
    return "\n".join(parts)


def extract_class(raw: str, valid_classes: list[str]) -> str | None:
    try:
        data = json.loads(raw)
        p_class = data.get("planet_classification", "").strip()
        if p_class in valid_classes:
            return p_class
    except (json.JSONDecodeError, AttributeError):
        pass

    for vc in valid_classes:
        if vc.lower() in raw.lower():
            return vc
    return None


def call_llm(client: OpenAI, model: str, obs) -> str:
    user_prompt = build_user_prompt(obs)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=64,
    )
    raw = response.choices[0].message.content.strip()
    p_class = extract_class(raw, obs.available_classifications)
    if p_class is None:
        raise ValueError(f"Could not parse class from LLM output: {raw}")
    return p_class


# ──────────────────────────────────────────────────────────────
#  Main loop
# ──────────────────────────────────────────────────────────────

def main():
    hf_token = os.environ.get("HF_TOKEN")
    api_base_url = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1/")
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

    if not hf_token:
        print("ERROR: HF_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(
        base_url=api_base_url,
        api_key=hf_token,
    )

    env = AstroEnvironment()

    for task_index in range(3):
        task_name = f"astro_task_{task_index}"
        
        obs = env.reset(episode_id=str(task_index))
        emit_start(task_name)

        step = 1
        try:
            p_class = call_llm(client, model_name, obs)
        except Exception as e:
            print(f"LLM error at task {task_name}: {e}", file=sys.stderr)
            emit_step(step, 0.05)
            emit_end(task_name=task_name, score=0.05, steps=step)
            continue

        action = Action(planet_classification=p_class)
        obs = env.step(action)
        reward = obs.reward

        emit_step(step, reward)
        emit_end(task_name=task_name, score=round(reward, 2), steps=step)


if __name__ == "__main__":
    main()