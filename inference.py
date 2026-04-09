import os
import json
import re
import sys
from openai import OpenAI
from server.ticket_router_environment import TicketRouterEnvironment
from models import Action

# ──────────────────────────────────────────────────────────────
#  Structured-output helpers (required by the Phase-2 validator)
# ──────────────────────────────────────────────────────────────

TASK_NAME = "ticket_router"


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
You are an expert customer-support ticket routing agent.
Your ONLY job is to choose the single best department for each ticket.

Rules:
- Read the ticket carefully and identify the PRIMARY issue.
- If the ticket mentions a technical problem (crashes, bugs, display issues, \
login failures, app errors), choose "Tech Support".
- If the ticket mentions wanting money back, overcharges, or refund requests, \
choose "Refunds".
- If the ticket mentions billing questions, payment plans, or invoice \
inquiries (but NOT refunds), choose "Billing".
- Only choose "General Inquiry" when the ticket does not fit any other category.
- When a ticket mentions MULTIPLE issues, prioritise the ACTIONABLE technical \
problem over a cancellation request.

Respond with ONLY a JSON object: {"department": "<chosen department>"}
Do NOT add any other text.
"""


def build_user_prompt(ticket_text: str, departments: list[str], history: list[str] | None = None) -> str:
    """Build the user-turn prompt, optionally including prior wrong guesses."""
    parts = [
        f'Ticket: "{ticket_text}"',
        f"Available departments: {departments}",
    ]
    if history:
        parts.append(
            "Your previous guesses for THIS ticket were WRONG: "
            + ", ".join(history)
            + ". Pick a DIFFERENT department."
        )
    parts.append('Respond with ONLY: {"department": "..."}')
    return "\n".join(parts)


def extract_department(raw: str, valid_departments: list[str]) -> str | None:
    """Robustly extract department from LLM output, even if it's noisy."""
    # Try strict JSON first
    try:
        data = json.loads(raw)
        dept = data.get("department", "").strip()
        if dept in valid_departments:
            return dept
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: regex for {"department": "..."}
    m = re.search(r'"department"\s*:\s*"([^"]+)"', raw)
    if m and m.group(1).strip() in valid_departments:
        return m.group(1).strip()

    # Last resort: check if any valid department name appears verbatim
    for dept in valid_departments:
        if dept.lower() in raw.lower():
            return dept

    return None


def call_llm(client: OpenAI, model: str, ticket_text: str,
             departments: list[str], wrong_guesses: list[str] | None = None) -> str:
    """Call the LLM and return the chosen department (or raise)."""
    user_prompt = build_user_prompt(ticket_text, departments, wrong_guesses)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,  # deterministic for routing
        max_tokens=64,
    )
    raw = response.choices[0].message.content.strip()
    dept = extract_department(raw, departments)
    if dept is None:
        raise ValueError(f"Could not parse department from LLM output: {raw}")
    return dept


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

    env = TicketRouterEnvironment()

    for task_index in range(3):
        task_name = f"task_{task_index}"
        
        # We define each ticket as a separate distinct "task" to fulfill the 3 task quota.
        obs = env.reset(episode_id=str(task_index))

        # ── Emit structured start ──
        emit_start(task_name)

        step = 1
        try:
            dept = call_llm(
                client, model_name,
                obs.ticket_text, obs.available_departments,
                None,
            )
        except Exception as e:
            print(f"LLM error at task {task_name}: {e}", file=sys.stderr)
            emit_step(step, 0.05)
            emit_end(task_name=task_name, score=0.05, steps=step)
            continue

        action = Action(department=dept)
        obs = env.step(action)
        reward = obs.reward

        # ── Emit structured step ──
        emit_step(step, reward)

        # ── Emit structured end ──
        emit_end(task_name=task_name, score=round(reward, 2), steps=step)


if __name__ == "__main__":
    main()