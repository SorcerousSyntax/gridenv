"""
GridWorld Graders — deterministic scoring for all 3 tasks.
All scores returned in [0.0, 1.0] with partial credit.
"""
from __future__ import annotations
from typing import Any, Dict


# ── EASY: Reach the Goal ───────────────────────────────────────────────────

class ReachGoalGrader:
    """
    Score = 1.0 if agent reaches goal.
    Partial credit for getting closer to goal.
    Step efficiency bonus for reaching faster.

    Components:
      - Goal reached (70%)
      - Proximity progress (20%) — how close agent got at best
      - Efficiency (10%) — steps saved vs max
    """

    def __init__(self, grid_size: int, max_steps: int):
        self.grid_size = grid_size
        self.max_steps = max_steps

    def score(self, state: Dict[str, Any]) -> Dict[str, Any]:
        won         = state.get("won", False)
        steps_used  = state.get("current_step", self.max_steps)
        agent_pos   = state.get("agent_pos", (0, 0))
        goal_pos    = state.get("goal_pos", (4, 4))
        best_dist   = state.get("best_distance_to_goal", None)

        # Manhattan distance: max possible = (size-1)*2
        max_dist = (self.grid_size - 1) * 2
        current_dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
        if best_dist is None:
            best_dist = current_dist

        goal_score = 1.0 if won else 0.0

        # Proximity: how close did agent ever get?
        proximity = 1.0 - (best_dist / max_dist) if max_dist > 0 else 0.0

        # Efficiency: bonus for finishing early
        if won:
            efficiency = 1.0 - (steps_used / self.max_steps)
        else:
            efficiency = 0.0

        final = 0.70 * goal_score + 0.20 * proximity + 0.10 * efficiency
        final = round(min(1.0, max(0.0, final)), 4)

        return {
            "score": final,
            "goal_reached": won,
            "proximity_score": round(proximity, 4),
            "efficiency_score": round(efficiency, 4),
            "feedback": (
                "Goal reached! Great job." if won
                else f"Did not reach goal. Best distance: {best_dist} cells away."
            )
        }


# ── MEDIUM: Collect and Escape ─────────────────────────────────────────────

class CollectAndEscapeGrader:
    """
    Score components:
      - Coins collected (40%) — fraction of total coins gathered
      - Goal reached (40%) — did agent escape?
      - Efficiency (20%) — steps saved

    Bonus: +0.1 if agent collected ALL coins before escaping (capped at 1.0)
    """

    def __init__(self, coins_total: int, max_steps: int, coins_required: int = 3):
        self.coins_total    = max(1, coins_total)
        self.max_steps      = max_steps
        self.coins_required = coins_required

    def score(self, state: Dict[str, Any]) -> Dict[str, Any]:
        won              = state.get("won", False)
        steps_used       = state.get("current_step", self.max_steps)
        coins_collected  = state.get("coins_collected", 0)

        coin_fraction = coins_collected / self.coins_total
        goal_score    = 1.0 if won else 0.0

        # Partial goal credit: reached exit but not enough coins
        met_requirement = coins_collected >= self.coins_required
        if won and not met_requirement:
            goal_score = 0.3   # reached exit but ignored coins

        efficiency = (1.0 - steps_used / self.max_steps) if won else 0.0

        final = 0.40 * coin_fraction + 0.40 * goal_score + 0.20 * efficiency

        # All-coins bonus
        if coins_collected == self.coins_total and won:
            final = min(1.0, final + 0.10)

        final = round(min(1.0, max(0.0, final)), 4)

        return {
            "score": final,
            "goal_reached": won,
            "coins_collected": coins_collected,
            "coins_total": self.coins_total,
            "coin_score": round(coin_fraction, 4),
            "efficiency_score": round(efficiency, 4),
            "feedback": (
                f"Escaped with {coins_collected}/{self.coins_total} coins!" if won
                else f"Did not escape. Collected {coins_collected}/{self.coins_total} coins."
            )
        }


# ── HARD: Survive and Escape ──────────────────────────────────────────────

class SurviveAndEscapeGrader:
    """
    Score components:
      - Escaped alive (35%)
      - Coins collected (30%) — fraction
      - Health remaining (15%) — health/max_health
      - Survival steps (10%) — how long before dying (if died)
      - Efficiency (10%) — steps saved if escaped

    Agents that die but collected coins still get partial credit.
    """

    def __init__(self, coins_total: int, max_steps: int, max_health: int = 3):
        self.coins_total = max(1, coins_total)
        self.max_steps   = max_steps
        self.max_health  = max_health

    def score(self, state: Dict[str, Any]) -> Dict[str, Any]:
        won             = state.get("won", False)
        steps_used      = state.get("current_step", self.max_steps)
        coins_collected = state.get("coins_collected", 0)
        health          = state.get("health", 0)
        alive           = health > 0

        escape_score  = 1.0 if won else 0.0
        coin_fraction = coins_collected / self.coins_total
        health_score  = health / self.max_health if alive else 0.0

        # Survival credit even if died
        survival_score = steps_used / self.max_steps if not won else 1.0

        efficiency = (1.0 - steps_used / self.max_steps) if won else 0.0

        final = (
            0.35 * escape_score  +
            0.30 * coin_fraction +
            0.15 * health_score  +
            0.10 * survival_score +
            0.10 * efficiency
        )
        final = round(min(1.0, max(0.0, final)), 4)

        if won:
            feedback = f"Survived and escaped! Health: {health}/3, Coins: {coins_collected}/{self.coins_total}"
        elif not alive:
            feedback = f"Agent died. Coins collected: {coins_collected}/{self.coins_total}"
        else:
            feedback = f"Time limit reached. Coins: {coins_collected}/{self.coins_total}, Health: {health}/3"

        return {
            "score": final,
            "goal_reached": won,
            "survived": alive,
            "coins_collected": coins_collected,
            "coins_total": self.coins_total,
            "health_remaining": health,
            "escape_score": round(escape_score, 4),
            "coin_score": round(coin_fraction, 4),
            "health_score": round(health_score, 4),
            "feedback": feedback
        }
