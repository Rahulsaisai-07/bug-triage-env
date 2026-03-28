"""
Baseline agent for Bug Triage Environment.
Uses keyword heuristics to solve each task type.
Run against your deployed Space:
  python baseline.py --url https://rahulsai07-bug-triage-env.hf.space
"""
import argparse
import requests
import random

def classify_severity(bug_report: str) -> str:
    report = bug_report.lower()
    if any(w in report for w in ["production", "down", "outage", "all users", "revenue", "500 error"]):
        return "critical"
    elif any(w in report for w in ["slow", "performance", "timeout", "spike", "30%"]):
        return "high"
    elif any(w in report for w in ["safari", "one browser", "workaround", "minor"]):
        return "medium"
    else:
        return "low"

def route_team(bug_report: str) -> str:
    report = bug_report.lower()
    if any(w in report for w in ["password", "security", "vulnerability", "plain text", "exploit"]):
        return "security"
    elif any(w in report for w in ["cpu", "server", "backup", "memory", "disk", "infrastructure"]):
        return "infrastructure"
    elif any(w in report for w in ["api", "database", "backend", "payment", "query", "endpoint"]):
        return "backend"
    else:
        return "frontend"

def check_duplicate(bug_report: str) -> tuple:
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


def run_baseline(base_url: str, episodes: int = 10):
    results = {"severity": [], "routing": [], "deduplication": []}

    for i in range(episodes):
        # Reset
        r = requests.post(f"{base_url}/reset")
        data = r.json()
        obs = data["observation"]
        task_type = obs["task_type"]
        bug_report = obs["bug_report"]

        # Act
        action = {}
        if task_type == "severity":
            action = {"severity": classify_severity(bug_report)}
        elif task_type == "routing":
            action = {"team": route_team(bug_report)}
        elif task_type == "deduplication":
            is_dup, dup_id = check_duplicate(bug_report)
            action = {"is_duplicate": is_dup, "duplicate_id": dup_id}

        r = requests.post(f"{base_url}/step", json=action)
        result = r.json()
        reward = result["reward"]
        feedback = result["observation"]["feedback"]

        results[task_type].append(reward)
        print(f"Episode {i+1:2d} | {task_type:15s} | reward={reward:.1f} | {feedback[:60]}")

    print("\n=== BASELINE RESULTS ===")
    for task, rewards in results.items():
        if rewards:
            avg = sum(rewards) / len(rewards)
            print(f"{task:20s}: {avg:.2f} avg reward ({len(rewards)} episodes)")
    total = [r for rewards in results.values() for r in rewards]
    print(f"{'OVERALL':20s}: {sum(total)/len(total):.2f} avg reward")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:7860", help="Environment base URL")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    run_baseline(args.url, args.episodes)