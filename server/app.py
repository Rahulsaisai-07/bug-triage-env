import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from server.environment import BugTriageEnvironment

app = FastAPI(
    title="Bug Triage Environment",
    description="OpenEnv-compatible environment for AI agent bug report triage",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = BugTriageEnvironment()


class ActionRequest(BaseModel):
    severity: Optional[str] = ""
    team: Optional[str] = ""
    is_duplicate: Optional[bool] = False
    duplicate_id: Optional[str] = None
    task_type: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "healthy", "environment": "BugTriageEnv", "version": "1.0.0"}


@app.post("/reset")
def reset(request: ActionRequest = ActionRequest()):
    obs = env.reset(task_type=request.task_type)
    state = env.state()
    return {
        "observation": {
            "done": obs.done,
            "reward": obs.reward,
            "task_type": obs.task_type,
            "bug_report": obs.bug_report,
            "available_severities": obs.available_severities,
            "available_teams": obs.available_teams,
            "message": obs.message,
            "feedback": obs.feedback,
        },
        "reward": obs.reward,
        "done": obs.done,
        "state": {
            "episode_id": state.episode_id,
            "step_count": state.step_count,
            "task_type": state.task_type,
            "total_reward": state.total_reward,
        }
    }


@app.post("/step")
def step(request: ActionRequest):
    from models import BugTriageAction
    action = BugTriageAction(
        severity=request.severity or "",
        team=request.team or "",
        is_duplicate=request.is_duplicate or False,
        duplicate_id=request.duplicate_id,
    )
    obs = env.step(action)
    state = env.state()
    return {
        "observation": {
            "done": obs.done,
            "reward": obs.reward,
            "task_type": obs.task_type,
            "bug_report": obs.bug_report,
            "available_severities": obs.available_severities,
            "available_teams": obs.available_teams,
            "message": obs.message,
            "feedback": obs.feedback,
        },
        "reward": obs.reward,
        "done": obs.done,
        "state": {
            "episode_id": state.episode_id,
            "step_count": state.step_count,
            "task_type": state.task_type,
            "total_reward": state.total_reward,
        }
    }


@app.get("/state")
def state():
    s = env.state()
    return {
        "episode_id": s.episode_id,
        "step_count": s.step_count,
        "task_type": s.task_type,
        "total_reward": s.total_reward,
    }


@app.get("/")
def root():
    return {
        "name": "Bug Triage Environment",
        "description": "AI agent environment for triaging software bug reports",
        "tasks": ["severity classification", "team routing", "duplicate detection"],
        "endpoints": ["/reset", "/step", "/state", "/health", "/docs"]
    }


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
