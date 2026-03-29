"""
GridWorld Map Generator — procedural level creation for all 3 tasks.
"""
from __future__ import annotations
import random
from typing import List, Tuple, Dict, Any


EMPTY = "."
WALL  = "#"
AGENT = "A"
GOAL  = "G"
COIN  = "C"
TRAP  = "T"
ENEMY = "E"


def _empty_grid(rows: int, cols: int) -> List[List[str]]:
    return [[EMPTY] * cols for _ in range(rows)]


def _place(grid, r, c, symbol):
    grid[r][c] = symbol


def _random_empty(grid, rng, exclude=None):
    """Pick a random empty cell."""
    exclude = set(exclude or [])
    rows, cols = len(grid), len(grid[0])
    empties = [
        (r, c) for r in range(rows) for c in range(cols)
        if grid[r][c] == EMPTY and (r, c) not in exclude
    ]
    return rng.choice(empties)


def _add_walls(grid, rng, wall_count):
    rows, cols = len(grid), len(grid[0])
    placed = 0
    attempts = 0
    while placed < wall_count and attempts < wall_count * 10:
        attempts += 1
        r = rng.randint(1, rows - 2)
        c = rng.randint(1, cols - 2)
        if grid[r][c] == EMPTY:
            grid[r][c] = WALL
            placed += 1


def render_grid(grid: List[List[str]]) -> str:
    """ASCII render of the grid with a border."""
    rows, cols = len(grid), len(grid[0])
    top = "+" + "-" * (cols * 2 - 1) + "+"
    lines = [top]
    for row in grid:
        lines.append("|" + " ".join(row) + "|")
    lines.append(top)
    return "\n".join(lines)


# ── Task 1: Easy — 5x5, no hazards ────────────────────────────────────────

def make_easy_map(seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    size = 5
    grid = _empty_grid(size, size)

    # Fixed start top-left, goal bottom-right quadrant
    agent_pos = (0, 0)
    goal_pos  = (rng.randint(3, 4), rng.randint(3, 4))
    _place(grid, *agent_pos, AGENT)
    _place(grid, *goal_pos, GOAL)

    # A few walls for mild interest
    _add_walls(grid, rng, wall_count=3)
    # Make sure start/goal not overwritten
    _place(grid, *agent_pos, AGENT)
    _place(grid, *goal_pos, GOAL)

    return {
        "grid": grid,
        "agent_pos": agent_pos,
        "goal_pos": goal_pos,
        "coin_positions": [],
        "trap_positions": [],
        "enemy_positions": [],
        "coins_total": 0,
        "health": 3,
        "size": size
    }


# ── Task 2: Medium — 7x7, coins required ──────────────────────────────────

def make_medium_map(seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    size = 7
    grid = _empty_grid(size, size)

    agent_pos = (0, 0)
    goal_pos  = (rng.randint(4, 6), rng.randint(4, 6))
    _place(grid, *agent_pos, AGENT)
    _place(grid, *goal_pos, GOAL)

    # Walls
    _add_walls(grid, rng, wall_count=8)
    _place(grid, *agent_pos, AGENT)
    _place(grid, *goal_pos, GOAL)

    # Place 5 coins
    coin_positions = []
    for _ in range(5):
        pos = _random_empty(grid, rng, exclude=[agent_pos, goal_pos] + coin_positions)
        _place(grid, *pos, COIN)
        coin_positions.append(pos)

    return {
        "grid": grid,
        "agent_pos": agent_pos,
        "goal_pos": goal_pos,
        "coin_positions": coin_positions,
        "trap_positions": [],
        "enemy_positions": [],
        "coins_total": len(coin_positions),
        "health": 3,
        "size": size
    }


# ── Task 3: Hard — 9x9, traps + enemies ───────────────────────────────────

def make_hard_map(seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    size = 9
    grid = _empty_grid(size, size)

    agent_pos = (0, 0)
    goal_pos  = (rng.randint(6, 8), rng.randint(6, 8))
    _place(grid, *agent_pos, AGENT)
    _place(grid, *goal_pos, GOAL)

    # Walls
    _add_walls(grid, rng, wall_count=15)
    _place(grid, *agent_pos, AGENT)
    _place(grid, *goal_pos, GOAL)

    occupied = [agent_pos, goal_pos]

    # 6 coins
    coin_positions = []
    for _ in range(6):
        pos = _random_empty(grid, rng, exclude=occupied + coin_positions)
        _place(grid, *pos, COIN)
        coin_positions.append(pos)

    # 4 traps
    trap_positions = []
    for _ in range(4):
        pos = _random_empty(grid, rng, exclude=occupied + coin_positions + trap_positions)
        _place(grid, *pos, TRAP)
        trap_positions.append(pos)

    # 3 enemies (start positions — they move each step)
    enemy_positions = []
    for _ in range(3):
        pos = _random_empty(grid, rng,
                            exclude=occupied + coin_positions + trap_positions + enemy_positions)
        _place(grid, *pos, ENEMY)
        enemy_positions.append(pos)

    return {
        "grid": grid,
        "agent_pos": agent_pos,
        "goal_pos": goal_pos,
        "coin_positions": coin_positions,
        "trap_positions": trap_positions,
        "enemy_positions": enemy_positions,
        "coins_total": len(coin_positions),
        "health": 3,
        "size": size
    }


MAP_MAKERS = {
    "reach_goal":         make_easy_map,
    "collect_and_escape": make_medium_map,
    "survive_and_escape": make_hard_map,
}
