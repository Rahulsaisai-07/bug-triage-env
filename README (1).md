---
title: Bug Triage Env
emoji: 🐛
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
---

## Why This Environment?

Every software company processes hundreds of bug reports weekly. Misclassified bugs cause:
- Critical issues getting deprioritized
- Wrong teams wasting time on irrelevant tickets
- Duplicate work from missed duplicate reports

A well-trained agent could reduce triage time by 60–80% while improving accuracy.

---

## Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| `severity_classification` | Easy | Classify bug as critical / high / medium / low |
| `team_assignment` | Medium | Classify severity + assign to correct team |
| `duplicate_detection` | Hard | Classify severity + assign team + detect duplicates |

---

## Action Space

```json
{
  "severity": "critical | high | medium | low",
  "team": "frontend | backend | infrastructure | security | mobile",
  "is_duplicate": true or false,
  "reasoning": "optional explanation"
}
```

## Observation Space

```json
{
  "done": true,
  "reward": 0.85,
  "bug_report": {
    "id": "BUG-004",
    "title": "Payment processing times out",
    "description": "...",
    "product_area": "payments",
    "error_logs": "...",
    "steps_to_reproduce": "..."
  },
  "feedback": "Score: 0.85/1.00\nSeverity correct...",
  "score_breakdown": {
    "severity": 1.0,
    "team": 0.6,
    "duplicate": 1.0
  },
  "attempts_remaining": 0,
  "current_task": "team_assignment"
}
```

---

## Reward Function

Rewards are partial — agents get credit for being close:

| Dimension | Weight (easy/medium/hard) | Scoring |
|-----------|--------------------------|---------|
| Severity | 100% / 40% / 30% | 1.0 exact, 0.5 one level off, 0.0 otherwise |
| Team | — / 60% / 40% | 1.0 exact, 0.0 wrong |
| Duplicate | — / — / 30% | 1.0 correct, 0.0 wrong |

---

## Setup & Usage

### Local

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t bug-triage-env .
docker run -p 8000:8000 bug-triage-env
```

### Quick test

```python
import requests

# Reset (start new episode)
obs = requests.post("http://localhost:8000/reset",
                    params={"task": "severity_classification"}).json()
print(obs["bug_report"]["title"])

# Step (submit triage decision)
result = requests.post("http://localhost:8000/step", json={
    "severity": "critical",
    "team": "backend",
    "is_duplicate": False,
    "task": "severity_classification"
}).json()
print(result["feedback"])
print(f"Reward: {result['reward']}")
```

---

## Baseline Scores

```bash
export OPENAI_API_KEY=your_key
python baseline.py
```

Expected results with GPT-4o-mini:

| Task | Baseline Score |
|------|---------------|
| severity_classification | ~0.72 |
| team_assignment | ~0.61 |
| duplicate_detection | ~0.48 |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Start new episode |
| `/step` | POST | Submit triage action |
| `/state` | GET | Current episode state |
| `/tasks` | GET | List tasks + action schema |
| `/baseline` | GET | Run heuristic baseline |
