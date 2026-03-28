"""
inference.py - Baseline inference script for Bug Triage Environment
Demonstrates an agent interacting with the environment.
"""
import requests
import sys

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"


def classify_severity(bug_report: str) -> str:
    report = bug_report.lower()
    if any(w in report for w in ["production", "down", "outage", "all users", "revenue", "500 error"]):
        return "critical"
    elif any(w in report for w in ["slow", "performance", "timeout", "spike", "30%"]):
        return "high"
    elif any(w in report for w in ["safari", "one browser", "workaround"]):
        return "medium"
    return "low"


def route_team(bug_report: str) -> str:
    report = bug_report.lower()
    if any(w in report for w in ["password", "security", "vulnerability", "plain text"]):
        return "security"
    elif any(w in report for w in ["cpu", "server", "backup", "memory", "infrastructure"]):
        return "infrastructure"
    elif any(w in report for w in ["api", "database", "payment", "query", "endpoint"]):
        return "backend"
    return "frontend"


def check_duplicate(bug_report: str):
    lines = bug_report.split("\n")
    new_report = ""
    existing = []
    for line in lines:
        if line.startswith("NEW REPORT:"):
            new_report = line.replace("NEW REPORT:", "").strip()
        elif line.strip().startswith("[BUG-"):
            parts = line.strip().split("]", 1)
            if len(parts) == 2:
                bug_id = parts[0].strip("[").strip()
                text = parts[1].strip()
                existing.append((bug_id, text))
    new_words = set(new_report.lower().split())
    best_match = None
    best_score = 0
    for bug_id, text in existing:
        words = set(text.lower().split())
        overlap = len(new_words & words) / max(len(new_words | words), 1)
        if overlap > best_score:
            best_score = overlap
            best_match = bug_id
    is_dup = best_score > 0.3
    return is_dup, best_match if is_dup else None


def run(episodes: int = 5):
    print(f"Running inference against: {BASE_URL}\n")
    total_reward = 0.0

    for i in range(episodes):
        r = requests.post(f"{BASE_URL}/reset")
        obs = r.json()["observation"]
        task_type = obs["task_type"]
        bug_report = obs["bug_report"]

        action = {}
        if task_type == "severity":
            action = {"severity": classify_severity(bug_report)}
        elif task_type == "routing":
            action = {"team": route_team(bug_report)}
        elif task_type == "deduplication":
            is_dup, dup_id = check_duplicate(bug_report)
            action = {"is_duplicate": is_dup, "duplicate_id": dup_id}

        r = requests.post(f"{BASE_URL}/step", json=action)
        result = r.json()
        reward = result["reward"]
        feedback = result["observation"]["feedback"]
        total_reward += reward

        print(f"Episode {i+1} | Task: {task_type:15s} | Reward: {reward:.1f} | {feedback[:60]}")

    print(f"\nAverage reward: {total_reward/episodes:.2f}")


if __name__ == "__main__":
    run()
