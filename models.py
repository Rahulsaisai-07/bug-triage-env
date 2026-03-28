from dataclasses import dataclass, field
from typing import Optional, List


class Action:
    pass

class Observation:
    pass

class State:
    pass


@dataclass
class BugTriageAction(Action):
    severity: str = ""
    team: str = ""
    is_duplicate: bool = False
    duplicate_id: Optional[str] = None


@dataclass
class BugTriageObservation(Observation):
    done: bool = False
    reward: float = 0.0
    task_type: str = ""
    bug_report: str = ""
    available_severities: List[str] = field(default_factory=lambda: ["critical", "high", "medium", "low"])
    available_teams: List[str] = field(default_factory=lambda: ["frontend", "backend", "infrastructure", "security"])
    message: str = ""
    feedback: str = ""


@dataclass
class BugTriageState(State):
    episode_id: str = ""
    step_count: int = 0
    task_type: str = ""
    total_reward: float = 0.0
