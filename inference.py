"""
GridWorld Survival — Inference Script
======================================
Runs an LLM agent against all 3 GridWorld tasks.

Required environment variables:
    API_BASE_URL   e.g. https://router.huggingface.co/v1
    MODEL_NAME     e.g. mistralai/Mistral-7B-Instruct-v0.3
    HF_TOKEN       Your Hugging Face token

Usage:
    export API_BASE_URL=https://router.huggingface.co/v1
    export MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3
    export HF_TOKEN=hf_...
    python inference.py

    # Rule-based (no API key needed):
    python inference.py --rule-based
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
from gridenv import GridWorldEnv

# ── Config ─────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN     = os.getenv("HF_TOKEN")     or os.getenv("API_KEY")
MAX_STEPS    = 60
TEMPERATURE  = 0.2
MAX_TOKENS   = 50
SEED         = 42

SYSTEM_PROMPT = """You are an AI agent playing a GridWorld navigation game.

You see a 2D ASCII grid:
  A = you (agent)
  G = goal (exit) — reach this to win
  C = coin — collect these for bonus score
  T = trap — costs 1 health point
  E = enemy — moves toward you, costs 1 health point
  # = wall — cannot pass through
  . = empty space

Respond with EXACTLY one action from: UP, DOWN, LEFT, RIGHT, STAY
No explanation. Just the action word."""


def build_prompt(obs: Dict) -> str:
    grid_text = obs.get("grid_text", "")
    agent_pos = obs.get("agent_pos", [0, 0])
    goal_pos  = obs.get("goal_pos",  [0, 0])
    coins     = obs.get("coins_collected", 0)
    total     = obs.get("coins_total", 0)
    health    = obs.get("health", 3)
    step      = obs.get("current_step", 0)
    max_steps = obs.get("max_steps", 30)
    hints     = obs.get("hints", [])
    history   = obs.get("history", [])[-3:]  # last 3 moves

    hist_text = ""
    if history:
        hist_text = "Recent moves: " + " → ".join(h["action"] for h in history)

    hint_text = ""
    if hints:
        hint_text = "HINT: " + " | ".join(hints)

    return f"""Grid:
{grid_text}

Agent at {agent_pos} | Goal at {goal_pos}
Health: {health}/3 | Coins: {coins}/{total} | Step: {step}/{max_steps}
{hist_text}
{hint_text}

Your action (UP/DOWN/LEFT/RIGHT/STAY):"""


# ── LLM Agent ──────────────────────────────────────────────────────────────

class LLMAgent:
    def __init__(self):
        if not HF_TOKEN:
            raise EnvironmentError("HF_TOKEN not set.")
        self.client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN
        )

    def act(self, obs: Dict) -> str:
        prompt = build_prompt(obs)
        try:
            resp = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt}
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            raw = resp.choices[0].message.content.strip().upper()
            for word in ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]:
                if word in raw:
                    return word
            return "STAY"
        except Exception as e:
            print(f"  [LLM Error] {e}")
            return "STAY"


# ── Rule-based Agent ───────────────────────────────────────────────────────

class RuleBasedAgent:
    """
    Simple greedy agent: move toward nearest coin first, then goal.
    Tries to avoid traps and enemies. No lookahead.
    """

    def act(self, obs: Dict) -> str:
        agent   = tuple(obs["agent_pos"])
        goal    = tuple(obs["goal_pos"])
        grid    = obs["grid"]
        coins   = obs.get("coins_remaining", 0)
        enemies = [tuple(e) for e in obs.get("enemies", [])]

        rows, cols = len(grid), len(grid[0])

        # Find nearest coin if coins remain
        target = None
        if coins > 0:
            best_dist = 9999
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == "C":
                        d = abs(r - agent[0]) + abs(c - agent[1])
                        if d < best_dist:
                            best_dist = d
                            target = (r, c)
        if target is None:
            target = goal

        # Move toward target, prefer safe cells
        ar, ac = agent
        tr, tc = target

        options = []
        if tr < ar: options.append(("UP",    (ar-1, ac)))
        if tr > ar: options.append(("DOWN",  (ar+1, ac)))
        if tc < ac: options.append(("LEFT",  (ar, ac-1)))
        if tc > ac: options.append(("RIGHT", (ar, ac+1)))
        options.append(("STAY", (ar, ac)))

        for action, (nr, nc) in options:
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            cell = grid[nr][nc]
            if cell == "#":
                continue
            if (nr, nc) in enemies:
                continue
            if cell == "T":
                # Only step on trap if no other option
                continue
            return action

        # Fallback: any non-wall move
        for action, (nr, nc) in options:
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != "#":
                return action

        return "STAY"


# ── Episode runner ─────────────────────────────────────────────────────────

def run_episode(task_id: str, agent, seed: int) -> Dict[str, Any]:
    env  = GridWorldEnv(task_id=task_id, seed=seed)
    obs  = env.reset()
    print(f"\n  Map preview:\n{obs.grid_text}")

    final_score = 0.0
    steps_used  = 0

    done = False
    while not done:
        obs_dict = obs.dict()
        action   = agent.act(obs_dict)

        obs, reward, done, info = env.step(action)
        steps_used = info["step"]
        final_score = reward.score

        print(
            f"[STEP] step={steps_used} reward={reward.delta:.4f}",
            flush=True,
        )

        status = f"[{action:5s}] score={reward.score:.4f} Δ={reward.delta:+.4f}"
        if info.get("won"):
            status += " ✓ WON"
        print(f"  Step {steps_used:3d}: {status}")

        if done:
            print(f"  → Terminal: {reward.feedback}")
            print(
                f"[END] task={task_id} score={final_score:.4f} steps={steps_used}",
                flush=True,
            )

    return {
        "task_id":     task_id,
        "final_score": final_score,
        "steps_used":  steps_used,
        "won":         info.get("won", False),
        "coins":       info.get("coins", 0),
        "seed":        seed,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GridWorld Baseline Inference")
    parser.add_argument("--rule-based", action="store_true",
                        help="Use rule-based agent (no API key needed)")
    parser.add_argument("--seed",  type=int, default=SEED)
    parser.add_argument("--tasks", nargs="+",
                        default=["reach_goal", "collect_and_escape", "survive_and_escape"])
    args = parser.parse_args()

    tasks   = args.tasks
    results = []

    print("=" * 60)
    print("GridWorld Survival — Baseline Evaluation")
    print(f"Model: {'rule-based' if args.rule_based else MODEL_NAME}")
    print(f"Seed:  {args.seed}")
    print("=" * 60)

    for task_id in tasks:
        diff = {"reach_goal": "EASY", "collect_and_escape": "MEDIUM", "survive_and_escape": "HARD"}
        print(f"\n{'─'*60}")
        print(f"Task: {task_id}  [{diff.get(task_id,'?')}]")
        print(f"{'─'*60}")

        # Emit START before any setup so validators always see structured output.
        print(f"[START] task={task_id}", flush=True)

        if args.rule_based:
            agent = RuleBasedAgent()
        else:
            try:
                agent = LLMAgent()
            except Exception as e:
                print(f"  ⚠ LLM unavailable ({e}), using rule-based fallback.")
                agent = RuleBasedAgent()

        try:
            result = run_episode(task_id, agent, seed=args.seed)
            results.append(result)
        except Exception as e:
            # Fallback structured output so parser still receives STEP/END blocks.
            print(f"[STEP] step=0 reward=0.0000", flush=True)
            print(f"[END] task={task_id} score=0.0000 steps=0", flush=True)
            print(f"  ⚠ Task failed: {e}")
            results.append({
                "task_id": task_id,
                "final_score": 0.0,
                "steps_used": 0,
                "won": False,
                "coins": 0,
                "seed": args.seed,
            })
        time.sleep(0.3)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Task':<30} {'Score':>7} {'Steps':>7} {'Won':>5}")
    print(f"{'─'*55}")
    for r in results:
        won_str = "✓" if r["won"] else "✗"
        print(f"{r['task_id']:<30} {r['final_score']:>7.4f} {r['steps_used']:>7} {won_str:>5}")

    avg = sum(r["final_score"] for r in results) / len(results)
    print(f"{'─'*55}")
    print(f"{'Average':<30} {avg:>7.4f}")

    # Save
    out = {
        "model":         "rule_based" if args.rule_based else MODEL_NAME,
        "api_base_url":  API_BASE_URL,
        "seed":          args.seed,
        "results":       results,
        "average_score": round(avg, 4),
    }
    with open("baseline_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → baseline_results.json")


if __name__ == "__main__":
    main()
