"""
main.py - Digital Forensics Lab FastAPI Server

Endpoints:
  POST /reset   Reset environment for a task
  POST /step    Take one action  (task passed in request body — consistent with /reset)
  GET  /state   Get current state
  GET  /health  Health check (required for HF Spaces)
  GET  /        Environment info
"""

import threading
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

from env import DigitalForensicsEnv, CaseFileState, ForensicAction, VALID_TASKS

app = FastAPI(
    title="Digital Forensics Lab",
    description="OpenEnv — AI agent as Enterprise Threat Intelligence Analyst.",
    version="1.0.0",
)

# Per-task environment store and per-task locks.
# Lock per task prevents concurrent requests to the same task from
# corrupting shared state.  For single-agent evaluation this is sufficient;
# for multi-agent use a proper session-ID based store.
_envs: Dict[str, DigitalForensicsEnv] = {}
_locks: Dict[str, threading.Lock] = {task: threading.Lock() for task in VALID_TASKS}


# ── Request / Response Models ────────────────────────

class ResetRequest(BaseModel):
    task: Optional[str] = "task_easy"


class StepRequest(BaseModel):
    """
    task is part of the request body (consistent with /reset).
    Previously task was a query param — this was an inconsistency.
    """
    task: str = "task_easy"
    action: str
    params: Dict[str, Any] = {}


# ── Endpoints ────────────────────────────────────────

@app.get("/")
def root() -> Dict:
    return {
        "name": "digital-forensics-lab",
        "version": "1.0.0",
        "tasks": VALID_TASKS,
        "description": (
            "Multimodal forensics env — investigate AI-generated content "
            "across code, audio, image, video, and hybrid cases."
        ),
    }


@app.get("/health")
def health() -> Dict:
    """Required for HF Spaces ping validation."""
    return {"status": "ok"}


@app.post("/reset")
def reset(request: Optional[ResetRequest] = None) -> Dict:
    """
    Reset environment for a given task.
    Body is optional — defaults to task_easy (required for openenv validate).
    Returns the initial CaseFileState observation directly.
    """
    task = request.task if request and request.task else "task_easy"
    if task not in VALID_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task '{task}'. Choose from {VALID_TASKS}",
        )

    with _locks[task]:
        env = DigitalForensicsEnv(task)
        obs = env.reset()
        _envs[task] = env

    return obs.model_dump()


@app.post("/step")
def step(request: StepRequest) -> Dict:
    """
    Take one action in the environment.
    Task is specified in the request body (consistent with /reset).
    """
    task = request.task
    if task not in VALID_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task '{task}'. Choose from {VALID_TASKS}",
        )

    with _locks[task]:
        env = _envs.get(task)
        if env is None:
            raise HTTPException(
                status_code=400,
                detail=f"Task '{task}' not initialised. POST /reset first.",
            )

        typed_action = ForensicAction(action=request.action, params=request.params)
        result = env.step(typed_action)

    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "reward_detail": result.reward_detail.model_dump(),
        "done": result.done,
        "info": result.info,
    }


@app.get("/state")
def get_state(task: str = "task_easy") -> Dict:
    """Return current state without advancing the episode."""
    if task not in VALID_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task '{task}'. Choose from {VALID_TASKS}",
        )

    with _locks.get(task, threading.Lock()):
        env = _envs.get(task)
        if env is None:
            raise HTTPException(
                status_code=400,
                detail=f"Task '{task}' not initialised. POST /reset first.",
            )
        return env.state().model_dump()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)
