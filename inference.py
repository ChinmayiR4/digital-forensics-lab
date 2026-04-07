"""
inference.py - Digital Forensics Lab Baseline Agent

Calls the deployed FastAPI server via HTTP (httpx) — tests the actual
deployment code path, not a local Python import.

MANDATORY env vars:
  HF_TOKEN      Required — your HuggingFace / API key (no default)
  API_BASE_URL  Default: https://router.huggingface.co/v1
  MODEL_NAME    Default: Qwen/Qwen2.5-72B-Instruct
  SERVER_URL    Default: http://localhost:7860
                Set to your HF Space URL for deployment evaluation:
                SERVER_URL=https://YOUR_USERNAME-digital-forensics-lab.hf.space

Usage:
  # Local (start server first: uvicorn main:app --port 7860)
  python inference.py

  # Against deployed HF Space:
  SERVER_URL=https://yourname-digital-forensics-lab.hf.space python inference.py

Output format (automated parsing — do not change):
  [START] task=<name> env=digital-forensics-lab model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

import asyncio
import json
import os
import sys

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Required environment variables ────────────────────
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: str | None = os.getenv("HF_TOKEN")
SERVER_URL: str = os.getenv("SERVER_URL", "http://localhost:7860").rstrip("/")

if HF_TOKEN is None:
    raise ValueError(
        "HF_TOKEN is missing.\n"
        "Add to your .env file:\n"
        "  HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx\n"
        "Get token at: https://huggingface.co/settings/tokens"
    )

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

ENV_NAME = "digital-forensics-lab"

# Per-task max steps — matches openenv.yaml max_steps field
TASK_MAX_STEPS: dict[str, int] = {
    "task_easy":   10,
    "task_medium": 10,
    "task_hard":   12,
    "task_expert": 10,
    "task_human":   8,
}

# Valid tasks — in difficulty order
VALID_TASKS = list(TASK_MAX_STEPS.keys())

# Score normalisation: theoretical maximum reward for a perfect episode.
# Perfect run = n_critical_tools * 0.50 + 1.00 (verdict).
# Using a generous global cap; clamped to [0, 1].
MAX_TOTAL_REWARD = 2.0

SYSTEM_PROMPT = """You are an Enterprise Threat Intelligence Analyst in a Digital Forensics Lab.
Investigate suspicious cases using forensic tools, build evidence, then submit a verdict.

=== AVAILABLE ACTIONS ===
Forensic Tools (each costs 0.15 budget per use — budget is LIMITED):
  analyze_metadata             - EXIF / file metadata
  reverse_image_search         - Earliest known date online (image/hybrid)
  text_perplexity_scan         - AI-text statistical score (text/code/hybrid)
  check_domain_age             - Website registration date (text/hybrid/audio)
  run_audio_frequency_scan     - Synthetic vocal artifacts (audio/hybrid/video)
  run_ast_code_analyzer        - LLM hallucination patterns in code (code)
  run_temporal_consistency_check - Deepfake artifacts in video (video/hybrid)

Final Action:
  submit_verdict               - Ends the episode

=== BUDGET WARNING ===
Budget starts at 1.0. Each tool costs 0.15. Running all 7 tools costs 1.05 — MORE
than your budget. You will be FORCED to end without submitting if you run out.
Maximum safe tool calls: 6. Choose wisely.

=== REWARD STRUCTURE ===
  +0.50  Critical tool used (the key evidence for this case)
  +0.20  Relevant tool used (valid for content_type, not the critical one)
  -0.20  Irrelevant tool used (wrong for content_type) — no evidence stored
  -0.40  Duplicate tool used (ran same tool twice) — AVOID
  -1.00  Budget depleted before verdict — CATASTROPHIC

=== VERDICT SCORING (final task_score) ===
  1.00  Correct verdict + ALL critical evidence cited
  0.75-0.99  Correct verdict + majority of critical evidence
  0.50  Correct verdict but zero evidence (lucky guess)
  0.10  False negative (missed real AI content)
  0.00  False positive (accused human of being AI) — WORST OUTCOME

=== OUTPUT FORMAT ===
Respond ONLY with valid JSON, no markdown:
{
  "action": "action_name",
  "params": {},
  "reasoning": "one sentence"
}

For submit_verdict:
{
  "action": "submit_verdict",
  "params": {
    "is_ai_generated": true,
    "confidence": 0.92,
    "critical_evidence_keys": ["tool_1", "tool_2"]
  },
  "reasoning": "one sentence"
}

List ALL tools you ran in critical_evidence_keys — not just one.
For all tool actions, params must be: {}
"""


def get_next_action(obs: dict, history: list) -> dict:
    """Call LLM to decide the next forensic action. Returns parsed JSON dict."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {
            "role": "user",
            "content": (
                f"Current case:\n{json.dumps(obs, indent=2)}\n\n"
                f"Next action? (JSON only)"
            ),
        },
    ]
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=512,
        temperature=0.0,
    )
    raw = (response.choices[0].message.content or "").strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
    return json.loads(raw)


async def run_episode(task: str) -> None:
    """
    Run one full episode against the FastAPI server via HTTP.
    The server must be reachable at SERVER_URL before calling this.
    """
    max_steps = TASK_MAX_STEPS.get(task, 10)

    async with httpx.AsyncClient(base_url=SERVER_URL, timeout=60.0) as http:

        # Reset episode
        reset_resp = await http.post("/reset", json={"task": task})
        reset_resp.raise_for_status()
        obs = reset_resp.json()  # /reset returns observation dict directly

        print(f"[START] task={task} env={ENV_NAME} model={MODEL_NAME}")
        sys.stdout.flush()

        history: list[dict] = []
        rewards: list[float] = []
        step_num = 0
        success = False
        task_score = 0.0

        for step_num in range(1, max_steps + 1):
            if obs.get("done", False):
                break

            last_error: str | None = None

            try:
                action_json = get_next_action(obs, history)
                action = str(action_json.get("action", "submit_verdict"))
                params = dict(action_json.get("params", {}))
                reasoning = str(action_json.get("reasoning", ""))
            except Exception as exc:
                last_error = str(exc)
                gathered = list(obs.get("gathered_evidence", {}).keys())
                action = "submit_verdict"
                params = {
                    "is_ai_generated": False,
                    "confidence": 0.5,
                    "critical_evidence_keys": gathered,
                }
                reasoning = f"API error — safe default: {gathered}"

            # Call step endpoint
            step_resp = await http.post(
                "/step",
                json={"task": task, "action": action, "params": params},
            )
            step_resp.raise_for_status()
            result = step_resp.json()

            reward = round(result.get("reward", 0.0), 2)
            done = result.get("done", False)
            info = result.get("info", {})
            obs = result.get("observation", obs)

            rewards.append(reward)

            action_str = (
                f"{action}({json.dumps(params, separators=(',', ':'))})"
                if params
                else f"{action}()"
            )
            action_str = action_str.replace("\n", " ").replace("\r", "")
            error_str = last_error.replace("\n", " ") if last_error else "null"
            done_str = "true" if done else "false"

            print(
                f"[STEP] step={step_num} action={action_str} "
                f"reward={reward:.2f} done={done_str} error={error_str}"
            )
            sys.stdout.flush()

            history.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"action": action, "params": params, "reasoning": reasoning}
                    ),
                }
            )
            history.append(
                {
                    "role": "user",
                    "content": (
                        f"Result — reward:{reward:.2f} done:{done_str} "
                        f"info:{json.dumps(info)}"
                    ),
                }
            )

            if done:
                task_score = float(info.get("task_score", 0.0))
                success = (
                    info.get("result", "").startswith("CORRECT_")
                    and info.get("result") != "CORRECT_NO_EVIDENCE"
                )
                break

        # Normalised cumulative score (clamped to [0, 1])
        score = min(
            max(sum(rewards) / MAX_TOTAL_REWARD, 0.0),
            1.0,
        )
        # Override with grader task_score when available (more precise)
        if task_score > 0.0:
            score = task_score

        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        success_str = "true" if success else "false"
        print(
            f"[END] success={success_str} steps={step_num} "
            f"score={score:.2f} rewards={rewards_str}"
        )
        sys.stdout.flush()


async def main() -> None:
    """Verify server is reachable, then run all tasks sequentially."""
    print(f"[DEBUG] Connecting to server at {SERVER_URL}", flush=True)
    async with httpx.AsyncClient(base_url=SERVER_URL, timeout=15.0) as http:
        try:
            health = await http.get("/health")
            health.raise_for_status()
        except Exception as exc:
            print(
                f"[ERROR] Server not reachable at {SERVER_URL}: {exc}\n"
                f"Start the server first:\n"
                f"  uvicorn main:app --host 0.0.0.0 --port 7860\n"
                f"Or set SERVER_URL to your HF Space URL.",
                flush=True,
            )
            sys.exit(1)

    for task in VALID_TASKS:
        await run_episode(task)
        print()  # blank line between tasks


if __name__ == "__main__":
    asyncio.run(main())
