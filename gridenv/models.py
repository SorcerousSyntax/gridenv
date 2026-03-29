"""
GridWorld Survival — Typed models for OpenEnv compliance.
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
from pydantic import BaseModel, Field
from enum import Enum


# ── Cell types ─────────────────────────────────────────────────────────────
class Cell(str, Enum):
    EMPTY  = "."
    WALL   = "#"
    AGENT  = "A"
    GOAL   = "G"
    COIN   = "C"
    TRAP   = "T"
    ENEMY  = "E"


# ── Actions ────────────────────────────────────────────────────────────────
class Action(str, Enum):
    UP    = "UP"
    DOWN  = "DOWN"
    LEFT  = "LEFT"
    RIGHT = "RIGHT"
    STAY  = "STAY"


# ── Observation ────────────────────────────────────────────────────────────
class HistoryEntry(BaseModel):
    step: int
    action: str
    reward: float
    feedback: str


class Observation(BaseModel):
    task_id: str
    task_description: str
    grid: List[List[str]]          # 2D grid as strings
    grid_text: str                 # Human-readable ASCII render
    agent_pos: Tuple[int, int]
    goal_pos: Tuple[int, int]
    coins_collected: int = 0
    coins_total: int = 0
    coins_remaining: int = 0
    health: int = 3
    enemies: List[Tuple[int, int]] = Field(default_factory=list)
    current_step: int = 0
    max_steps: int = 30
    done: bool = False
    history: List[HistoryEntry] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    hints: List[str] = Field(default_factory=list)


# ── Reward ─────────────────────────────────────────────────────────────────
class RewardBreakdown(BaseModel):
    goal_reached: float  = 0.0
    coins_bonus: float   = 0.0
    step_penalty: float  = 0.0
    trap_penalty: float  = 0.0
    enemy_penalty: float = 0.0
    survival_bonus: float = 0.0


class Reward(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    delta: float = 0.0
    breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)
    feedback: str = ""
    is_terminal: bool = False


# ── Full State ─────────────────────────────────────────────────────────────
class EpisodeState(BaseModel):
    task_id: str
    grid: List[List[str]]
    agent_pos: Tuple[int, int]
    goal_pos: Tuple[int, int]
    coin_positions: List[Tuple[int, int]]
    trap_positions: List[Tuple[int, int]]
    enemy_positions: List[Tuple[int, int]]
    coins_collected: int = 0
    coins_total: int = 0
    health: int = 3
    current_step: int = 0
    max_steps: int = 30
    done: bool = False
    won: bool = False
    cumulative_score: float = 0.0
    action_history: List[str] = Field(default_factory=list)
