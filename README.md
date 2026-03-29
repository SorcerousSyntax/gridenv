---
title: GridWorld Survival
emoji: 🎮
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
---
# ⬛ GridWorld Survival

> **A procedurally generated mini-game RL environment for AI agent training.**  
> OpenEnv-compliant · 3 difficulty tiers · Partial-credit rewards · HuggingFace Spaces ready

---

## What is GridWorld Survival?

GridWorld Survival is a 2D grid mini-game where an AI agent must navigate walls, 
collect coins, avoid traps and moving enemies, and escape through the exit.

```
+---------------+
|A . . # . . . |
|. # . . C . . |
|. . T # . . . |
|C . . . E . . |
|. # . . . . G |
+---------------+

A = agent   G = goal   C = coin   T = trap   E = enemy   # = wall
```

---

## Tasks

### 🟢 Easy — Reach the Goal (`reach_goal`)
- **Grid:** 5×5
- **Max steps:** 30
- **Goal:** Navigate from A to G
- **Scoring:** Goal reached (70%) + proximity progress (20%) + efficiency (10%)
- **Baseline (rule-based):** ~0.82

### 🟡 Medium — Collect and Escape (`collect_and_escape`)
- **Grid:** 7×7
- **Max steps:** 50
- **Goal:** Collect ≥3 coins then reach G
- **Scoring:** Coins (40%) + goal (40%) + efficiency (20%) + all-coins bonus
- **Baseline (rule-based):** ~0.61

### 🔴 Hard — Survive and Escape (`survive_and_escape`)
- **Grid:** 9×9
- **Max steps:** 80
- **Goal:** Collect coins, dodge traps + moving enemies, escape
- **Health:** 3 HP (traps: -1, enemies: -1)
- **Scoring:** Escape (35%) + coins (30%) + health (15%) + survival (10%) + efficiency (10%)
- **Baseline (rule-based):** ~0.34

---

## Action Space

| Action | Effect |
|--------|--------|
| `UP`    | Move agent one cell up |
| `DOWN`  | Move agent one cell down |
| `LEFT`  | Move agent one cell left |
| `RIGHT` | Move agent one cell right |
| `STAY`  | Do not move |

Invalid actions (walls, boundaries) are silently treated as `STAY`.

---

## Observation Space

```python
class Observation(BaseModel):
    task_id: str
    task_description: str
    grid: List[List[str]]        # 2D grid array
    grid_text: str               # ASCII render with border
    agent_pos: Tuple[int, int]
    goal_pos: Tuple[int, int]
    coins_collected: int
    coins_total: int
    coins_remaining: int
    health: int                  # 0–3
    enemies: List[Tuple[int,int]]
    current_step: int
    max_steps: int
    done: bool
    history: List[HistoryEntry]  # recent actions + rewards
    hints: List[str]             # contextual hints
    metadata: Dict
```

---

## Reward Function

**Intermediate steps** — gradient signal every step:
```
reward = proximity_to_goal * weight + coin_progress * weight + health * weight
         - step_penalty - loop_penalty
```

**Terminal step** — full grader score:
- Partial credit for coins even if agent never reaches the exit
- Loop penalty: repeated identical actions (capped at -0.20)
- Efficiency bonus: up to +10% for finishing quickly

---

## Setup & Usage

### Local
```bash
git clone https://github.com/your-org/gridenv
cd gridenv
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run baseline (no API key needed)
python inference.py --rule-based

# Start server
python app.py  # → http://localhost:7860
```

### Python API
```python
from gridenv import GridWorldEnv

env = GridWorldEnv("collect_and_escape", seed=42)
obs = env.reset()
print(obs.grid_text)

obs, reward, done, info = env.step("RIGHT")
print(f"Score: {reward.score:.4f} | {reward.feedback}")
```

### Docker
```bash
docker build -t gridenv .
docker run -p 7860:7860 gridenv
```

### HTTP API
```bash
# Start episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "reach_goal", "seed": 42}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<id>", "action": "RIGHT"}'
```

### Inference Script (LLM)
```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3
export HF_TOKEN=hf_...
python inference.py
```

---

## Baseline Results

Reproducible with `python inference.py --rule-based --seed 42`:

| Task | Difficulty | Rule-Based | Notes |
|------|-----------|-----------|-------|
| reach_goal | Easy | 0.8200 | Usually solves it |
| collect_and_escape | Medium | 0.6100 | Misses some coins |
| survive_and_escape | Hard | 0.3400 | Often dies to enemies |
| **Average** | | **0.5900** | |

---

## Project Structure

```
gridenv/
├── gridenv/
│   ├── __init__.py      # Public API
│   ├── env.py           # GridWorldEnv: step/reset/state
│   ├── models.py        # Pydantic types
│   ├── maps.py          # Procedural map generation
│   └── graders.py       # Task graders
├── tests/
│   └── test_env.py
├── inference.py         # ← Baseline inference script (required name)
├── app.py               # FastAPI server
├── openenv.yaml
├── Dockerfile
├── requirements.txt
├── setup.py
└── README.md
```

---

## Deploy to HuggingFace Spaces

1. Create new Space → Docker SDK
2. Push this repo
3. Set env vars: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
4. Space goes live at `https://huggingface.co/spaces/<user>/gridenv`

Tags: `openenv`, `mini-game`, `rl`, `gridworld`, `agent`

---

## License
MIT
