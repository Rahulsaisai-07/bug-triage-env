import random
import uuid
from typing import Optional
from models import BugTriageAction, BugTriageObservation, BugTriageState

# Bug report dataset with ground truth labels
BUG_REPORTS = {
    "severity": [
        {
            "report": "Production database is down. All users cannot log in. Revenue impact: $10,000/minute.",
            "correct_severity": "critical",
            "explanation": "Production outage with direct revenue impact = critical severity."
        },
        {
            "report": "The 'Export to CSV' button on the reports page doesn't work in Safari browser.",
            "correct_severity": "medium",
            "explanation": "Feature broken in one browser, workaround exists = medium severity."
        },
        {
            "report": "Typo in footer: 'Copywright' should be 'Copyright'.",
            "correct_severity": "low",
            "explanation": "Minor cosmetic issue with no functional impact = low severity."
        },
        {
            "report": "Authentication service returning 500 errors for 30% of login attempts.",
            "correct_severity": "critical",
            "explanation": "Auth failures affecting large percentage of users = critical."
        },
        {
            "report": "Dashboard charts take 8 seconds to load instead of the usual 2 seconds.",
            "correct_severity": "high",
            "explanation": "Significant performance degradation affecting all users = high severity."
        },
    ],
    "routing": [
        {
            "report": "Login button is misaligned on mobile devices. The button overlaps with the username field.",
            "correct_team": "frontend",
            "explanation": "UI layout issue on mobile = frontend team responsibility."
        },
        {
            "report": "Payment processing API is returning timeout errors after 30 seconds for large transactions.",
            "correct_team": "backend",
            "explanation": "API timeout issue in payment processing = backend team."
        },
        {
            "report": "Server CPU usage spikes to 100% every night at 2 AM during backup jobs.",
            "correct_team": "infrastructure",
            "explanation": "Server resource issue during scheduled jobs = infrastructure team."
        },
        {
            "report": "User passwords are being stored in plain text in the database.",
            "correct_team": "security",
            "explanation": "Password storage vulnerability = security team."
        },
        {
            "report": "The date picker component crashes when selecting dates before 1970.",
            "correct_team": "frontend",
            "explanation": "UI component crash = frontend team."
        },
    ],
    "deduplication": [
        {
            "new_report": "App crashes when I try to upload a profile picture larger than 5MB.",
            "existing_reports": [
                {"id": "BUG-001", "text": "Application throws error when uploading images over 5MB in size."},
                {"id": "BUG-002", "text": "Search results don't update when filtering by date range."},
                {"id": "BUG-003", "text": "Email notifications are not being sent for new comments."},
            ],
            "is_duplicate": True,
            "duplicate_of": "BUG-001",
            "explanation": "Both reports describe the same 5MB image upload failure."
        },
        {
            "new_report": "The notification badge count doesn't reset after reading all messages.",
            "existing_reports": [
                {"id": "BUG-004", "text": "Profile picture upload fails silently without error message."},
                {"id": "BUG-005", "text": "Dark mode toggle doesn't persist after page refresh."},
                {"id": "BUG-006", "text": "Two-factor authentication codes expire too quickly."},
            ],
            "is_duplicate": False,
            "duplicate_of": None,
            "explanation": "No existing report matches the notification badge issue."
        },
    ]
}

SEVERITY_SCORES = {
    ("critical", "critical"): 1.0,
    ("high", "high"): 1.0,
    ("medium", "medium"): 1.0,
    ("low", "low"): 1.0,
    ("critical", "high"): 0.5,
    ("high", "critical"): 0.5,
    ("high", "medium"): 0.5,
    ("medium", "high"): 0.5,
    ("medium", "low"): 0.5,
    ("low", "medium"): 0.5,
    ("critical", "medium"): 0.0,
    ("critical", "low"): 0.0,
    ("low", "critical"): 0.0,
    ("low", "high"): 0.0,
}


class BugTriageEnvironment:
    def __init__(self):
        self._episode_id = None
        self._step_count = 0
        self._task_type = None
        self._current_bug = None
        self._total_reward = 0.0
        self._done = False

    def reset(self, task_type: Optional[str] = None) -> BugTriageObservation:
        self._episode_id = str(uuid.uuid4())[:8]
        self._step_count = 0
        self._total_reward = 0.0
        self._done = False

        # Cycle through task types or pick randomly
        if task_type and task_type in BUG_REPORTS:
            self._task_type = task_type
        else:
            self._task_type = random.choice(["severity", "routing", "deduplication"])

        self._current_bug = random.choice(BUG_REPORTS[self._task_type])

        return self._build_observation(
            message=f"New bug triage task: {self._task_type.upper()}. Analyze the bug report and respond.",
            reward=0.0,
            done=False,
            feedback=""
        )

    def step(self, action: BugTriageAction) -> BugTriageObservation:
        if self._done:
            return self._build_observation(
                message="Episode already complete. Call reset() to start a new task.",
                reward=0.0,
                done=True,
                feedback=""
            )

        self._step_count += 1
        reward, feedback = self._grade(action)
        self._total_reward += reward
        self._done = True  # One bug per episode

        return self._build_observation(
            message="Task complete!" if reward >= 0.8 else "Task complete. Review feedback.",
            reward=reward,
            done=True,
            feedback=feedback
        )

    def state(self) -> BugTriageState:
        return BugTriageState(
            episode_id=self._episode_id or "",
            step_count=self._step_count,
            task_type=self._task_type or "",
            total_reward=self._total_reward
        )

    def _grade(self, action: BugTriageAction):
        bug = self._current_bug

        if self._task_type == "severity":
            correct = bug["correct_severity"]
            given = action.severity.lower().strip()
            reward = SEVERITY_SCORES.get((given, correct), 0.0)
            if reward == 1.0:
                feedback = f"✅ Correct! {bug['explanation']}"
            elif reward == 0.5:
                feedback = f"⚠️ Partially correct. You said '{given}', correct is '{correct}'. {bug['explanation']}"
            else:
                feedback = f"❌ Incorrect. You said '{given}', correct is '{correct}'. {bug['explanation']}"
            return reward, feedback

        elif self._task_type == "routing":
            correct = bug["correct_team"]
            given = action.team.lower().strip()
            if given == correct:
                reward = 1.0
                feedback = f"✅ Correct! {bug['explanation']}"
            else:
                reward = 0.0
                feedback = f"❌ Incorrect. You said '{given}', correct is '{correct}'. {bug['explanation']}"
            return reward, feedback

        elif self._task_type == "deduplication":
            correct_dup = bug["is_duplicate"]
            correct_id = bug["duplicate_of"]

            if action.is_duplicate == correct_dup:
                if correct_dup and action.duplicate_id == correct_id:
                    reward = 1.0
                    feedback = f"✅ Perfect! Correctly identified as duplicate of {correct_id}. {bug['explanation']}"
                elif correct_dup and action.duplicate_id != correct_id:
                    reward = 0.5
                    feedback = f"⚠️ Correctly identified as duplicate but wrong ID. Expected {correct_id}. {bug['explanation']}"
                else:
                    reward = 1.0
                    feedback = f"✅ Correct! This is not a duplicate. {bug['explanation']}"
            else:
                reward = 0.0
                feedback = f"❌ Wrong. is_duplicate should be {correct_dup}. {bug['explanation']}"
            return reward, feedback

        return 0.0, "Unknown task type."

    def _build_observation(self, message, reward, done, feedback) -> BugTriageObservation:
        bug = self._current_bug
        if not bug:
            return BugTriageObservation(done=done, reward=reward, message=message, feedback=feedback)

        if self._task_type == "deduplication":
            existing = bug.get("existing_reports", [])
            existing_text = "\n".join([f"  [{r['id']}] {r['text']}" for r in existing])
            bug_text = f"NEW REPORT: {bug['new_report']}\n\nEXISTING REPORTS:\n{existing_text}"
        else:
            bug_text = bug.get("report", "")

        return BugTriageObservation(
            done=done,
            reward=reward,
            task_type=self._task_type,
            bug_report=bug_text,
            message=message,
            feedback=feedback
        )
