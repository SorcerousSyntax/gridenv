"""
GridWorld Survival — Core OpenEnv environment.
Implements step() / reset() / state() with full OpenEnv compliance.
"""
from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    Action, Observation, Reward, RewardBreakdown,
    EpisodeState, HistoryEntry
)
from .maps import MAP_MAKERS, render_grid, EMPTY, WALL, AGENT, GOAL, COIN, TRAP, ENEMY
from .graders import ReachGoalGrader, CollectAndEscapeGrader, SurviveAndEscapeGrader


TASK_CONFIG = {
    "reach_goal": {
        "description": (
            "Navigate the 5x5 grid from A (agent) to G (goal). "
            "Walls (#) block movement. Reach G in as few steps as possible.\n"
            "Actions: UP, DOWN, LEFT, RIGHT, STAY"
        ),
        "max_steps": 30,
        "health": 3,
        "coins_required": 0,
    },
    "collect_and_escape": {
        "description": (
            "Navigate the 7x7 grid. Collect at least 3 coins (C) before "
            "reaching the exit (G). Walls (#) block movement. "
            "More coins + fewer steps = higher score.\n"
            "Actions: UP, DOWN, LEFT, RIGHT, STAY"
        ),
        "max_steps": 50,
        "health": 3,
        "coins_required": 3,
    },
    "survive_and_escape": {
        "description": (
            "Navigate the 9x9 grid. Collect coins (C), avoid traps (T) and "
            "moving enemies (E), and escape through the exit (G). "
            "You have 3 health points. Traps cost 1 HP, enemies cost 1 HP per contact. "
            "Die and the episode ends. Partial credit for coins collected.\n"
            "Actions: UP, DOWN, LEFT, RIGHT, STAY"
        ),
        "max_steps": 80,
        "health": 3,
        "coins_required": 0,
    }
}

MOVE_DELTAS = {
    "UP":    (-1,  0),
    "DOWN":  ( 1,  0),
    "LEFT":  ( 0, -1),
    "RIGHT": ( 0,  1),
    "STAY":  ( 0,  0),
}


class GridWorldEnv:
    """
    OpenEnv-compliant GridWorld Survival environment.

    Tasks:
      - reach_goal          (easy)   5x5 grid, navigate to exit
      - collect_and_escape  (medium) 7x7 grid, collect coins then exit
      - survive_and_escape  (hard)   9x9 grid, traps + enemies + coins + exit
    """

    def __init__(self, task_id: str = "reach_goal", seed: Optional[int] = None):
        if task_id not in TASK_CONFIG:
            raise ValueError(f"Unknown task '{task_id}'. Choose from {list(TASK_CONFIG)}")
        self.task_id  = task_id
        self.seed     = seed
        self._rng     = random.Random(seed)
        self._state: Optional[EpisodeState] = None
        self._grader  = None
        self._prev_score: float = 0.0
        self._best_distance: int = 9999
        self._repeated_actions: Dict[str, int] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset environment and return the initial observation."""
        cfg     = TASK_CONFIG[self.task_id]
        map_fn  = MAP_MAKERS[self.task_id]
        episode_seed = self._rng.randint(0, 99999)
        m = map_fn(episode_seed)

        self._state = EpisodeState(
            task_id          = self.task_id,
            grid             = copy.deepcopy(m["grid"]),
            agent_pos        = m["agent_pos"],
            goal_pos         = m["goal_pos"],
            coin_positions   = list(m["coin_positions"]),
            trap_positions   = list(m["trap_positions"]),
            enemy_positions  = list(m["enemy_positions"]),
            coins_collected  = 0,
            coins_total      = m["coins_total"],
            health           = m["health"],
            current_step     = 0,
            max_steps        = cfg["max_steps"],
            done             = False,
            won              = False,
            cumulative_score = 0.0,
            action_history   = []
        )

        self._grader = self._build_grader()
        self._prev_score = 0.0
        self._best_distance = self._manhattan(self._state.agent_pos, self._state.goal_pos)
        self._repeated_actions = {}

        return self._build_observation()

    def step(self, action: str) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one action.

        Args:
            action: One of UP / DOWN / LEFT / RIGHT / STAY

        Returns:
            observation, reward, done, info
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset().")

        # Normalise
        action = str(action).upper().strip()
        if action not in MOVE_DELTAS:
            action = "STAY"

        self._state.current_step += 1
        self._state.action_history.append(action)

        # Loop penalty tracking
        self._repeated_actions[action] = self._repeated_actions.get(action, 0) + 1

        # --- Move agent ---
        dr, dc = MOVE_DELTAS[action]
        r, c   = self._state.agent_pos
        nr, nc = r + dr, c + dc

        grid = self._state.grid
        rows, cols = len(grid), len(grid[0])

        feedback_parts = []

        # Boundary / wall check
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != WALL:
            # Clear old position
            grid[r][c] = EMPTY
            r, c = nr, nc

            cell = grid[r][c]

            # Collect coin
            if cell == COIN:
                self._state.coins_collected += 1
                if (r, c) in self._state.coin_positions:
                    self._state.coin_positions.remove((r, c))
                feedback_parts.append(f"Collected a coin! ({self._state.coins_collected}/{self._state.coins_total})")

            # Trap
            elif cell == TRAP:
                self._state.health -= 1
                feedback_parts.append(f"Hit a trap! Health: {self._state.health}/3")

            # Reached goal
            elif cell == GOAL:
                self._state.won  = True
                self._state.done = True
                feedback_parts.append("Reached the goal! Episode complete.")

            # Place agent
            if not self._state.done:
                grid[r][c] = AGENT
            self._state.agent_pos = (r, c)
        else:
            feedback_parts.append("Blocked by wall or boundary.")

        # --- Move enemies (hard task) ---
        if self.task_id == "survive_and_escape" and not self._state.done:
            self._move_enemies()
            # Check if enemy moved onto agent
            if self._state.agent_pos in self._state.enemy_positions:
                self._state.health -= 1
                feedback_parts.append(f"Enemy hit you! Health: {self._state.health}/3")

        # --- Death check ---
        if self._state.health <= 0:
            self._state.done = True
            feedback_parts.append("You died!")

        # --- Step limit ---
        if self._state.current_step >= self._state.max_steps and not self._state.done:
            self._state.done = True
            feedback_parts.append("Time limit reached.")

        # --- Update best distance ---
        dist = self._manhattan(self._state.agent_pos, self._state.goal_pos)
        if dist < self._best_distance:
            self._best_distance = dist

        # --- Compute reward ---
        reward = self._compute_reward(feedback_parts)
        self._state.cumulative_score = max(self._state.cumulative_score, reward.score)

        obs = self._build_observation()
        obs.history.append(HistoryEntry(
            step=self._state.current_step,
            action=action,
            reward=reward.score,
            feedback=reward.feedback
        ))
        obs.done = self._state.done

        info = {
            "task_id":   self.task_id,
            "step":      self._state.current_step,
            "won":       self._state.won,
            "health":    self._state.health,
            "coins":     self._state.coins_collected,
            "best_dist": self._best_distance,
        }
        return obs, reward, self._state.done, info

    def state(self) -> EpisodeState:
        """Return full internal state (for debugging / inspection)."""
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return copy.deepcopy(self._state)

    # ── Reward ─────────────────────────────────────────────────────────────

    def _compute_reward(self, feedback_parts: List[str]) -> Reward:
        s = self._state

        # Loop / repeat penalty
        total_actions = len(s.action_history)
        repeated = sum(max(0, v - 1) for v in self._repeated_actions.values())
        loop_penalty = min(0.2, repeated / max(1, total_actions) * 0.3)

        # Step penalty (small, encourages efficiency)
        step_penalty = s.current_step / s.max_steps * 0.05

        if s.done:
            grader_state = {
                "won":              s.won,
                "current_step":     s.current_step,
                "coins_collected":  s.coins_collected,
                "health":           s.health,
                "agent_pos":        s.agent_pos,
                "goal_pos":         s.goal_pos,
                "best_distance_to_goal": self._best_distance,
            }
            result = self._grader.score(grader_state)
            base   = result["score"]
            feedback = result["feedback"]
            is_terminal = True
        else:
            # Intermediate: partial progress signal
            dist     = self._manhattan(s.agent_pos, s.goal_pos)
            max_dist = (len(s.grid) - 1) * 2
            proximity = 1.0 - dist / max_dist if max_dist else 0.0

            coin_progress = (s.coins_collected / s.coins_total) if s.coins_total > 0 else 0.0
            health_ratio  = s.health / 3.0

            if self.task_id == "reach_goal":
                base = 0.8 * proximity + 0.2 * (1 - step_penalty)
            elif self.task_id == "collect_and_escape":
                base = 0.4 * coin_progress + 0.5 * proximity + 0.1 * (1 - step_penalty)
            else:
                base = 0.3 * coin_progress + 0.4 * proximity + 0.2 * health_ratio + 0.1 * (1 - step_penalty)

            feedback    = " | ".join(feedback_parts) if feedback_parts else "Moving..."
            is_terminal = False

        final = round(min(1.0, max(0.0, base - loop_penalty)), 4)
        delta = round(final - self._prev_score, 4)
        self._prev_score = final

        return Reward(
            score=final,
            delta=delta,
            breakdown=RewardBreakdown(
                goal_reached  = 1.0 if s.won else 0.0,
                coins_bonus   = s.coins_collected / max(1, s.coins_total),
                step_penalty  = round(step_penalty, 4),
                loop_penalty  = round(loop_penalty, 4),
                survival_bonus = s.health / 3.0
            ),
            feedback=feedback,
            is_terminal=is_terminal
        )

    # ── Enemy movement ─────────────────────────────────────────────────────

    def _move_enemies(self):
        """Move each enemy one step randomly (or toward agent 50% of the time)."""
        grid  = self._state.grid
        rows, cols = len(grid), len(grid[0])
        new_positions = []

        for (er, ec) in self._state.enemy_positions:
            grid[er][ec] = EMPTY

            ar, ac = self._state.agent_pos
            # 50% chance: move toward agent
            if self._rng.random() < 0.5:
                dr = 0 if er == ar else (1 if ar > er else -1)
                dc = 0 if ec == ac else (1 if ac > ec else -1)
                # Prefer one axis
                if dr != 0 and dc != 0:
                    if self._rng.random() < 0.5:
                        dc = 0
                    else:
                        dr = 0
            else:
                dr, dc = self._rng.choice([(-1,0),(1,0),(0,-1),(0,1)])

            nr, nc = er + dr, ec + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and grid[nr][nc] not in (WALL, ENEMY, GOAL)):
                new_positions.append((nr, nc))
                if grid[nr][nc] != AGENT:
                    grid[nr][nc] = ENEMY
            else:
                new_positions.append((er, ec))
                grid[er][ec] = ENEMY

        self._state.enemy_positions = new_positions

    # ── Helpers ────────────────────────────────────────────────────────────

    def _manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _build_grader(self):
        s = self._state
        if self.task_id == "reach_goal":
            return ReachGoalGrader(grid_size=len(s.grid), max_steps=s.max_steps)
        elif self.task_id == "collect_and_escape":
            return CollectAndEscapeGrader(
                coins_total=s.coins_total,
                max_steps=s.max_steps,
                coins_required=TASK_CONFIG[self.task_id]["coins_required"]
            )
        else:
            return SurviveAndEscapeGrader(
                coins_total=s.coins_total,
                max_steps=s.max_steps,
                max_health=s.health
            )

    def _build_observation(self) -> Observation:
        s = self._state
        hints = []
        if s.current_step > s.max_steps * 0.7:
            hints.append("Running low on steps — head for the exit!")
        if s.health == 1:
            hints.append("Critical health — avoid traps and enemies!")
        if self.task_id == "collect_and_escape" and s.coins_collected < 3:
            remaining = 3 - s.coins_collected
            hints.append(f"Need {remaining} more coin(s) before exiting.")

        return Observation(
            task_id          = self.task_id,
            task_description = TASK_CONFIG[self.task_id]["description"],
            grid             = copy.deepcopy(s.grid),
            grid_text        = render_grid(s.grid),
            agent_pos        = s.agent_pos,
            goal_pos         = s.goal_pos,
            coins_collected  = s.coins_collected,
            coins_total      = s.coins_total,
            coins_remaining  = len(s.coin_positions),
            health           = s.health,
            enemies          = list(s.enemy_positions),
            current_step     = s.current_step,
            max_steps        = s.max_steps,
            done             = s.done,
            history          = [],
            metadata         = {
                "task_id":  self.task_id,
                "won":      s.won,
                "seed":     self.seed,
            },
            hints=hints
        )
