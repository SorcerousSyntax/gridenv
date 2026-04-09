"""
GridWorld Survival — Test suite.
Run: pytest tests/ -v
"""
import pytest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from gridenv import GridWorldEnv


@pytest.mark.parametrize("task_id", ["reach_goal", "collect_and_escape", "survive_and_escape"])
def test_reset_returns_observation(task_id):
    env = GridWorldEnv(task_id, seed=1)
    obs = env.reset()
    assert obs.task_id == task_id
    assert len(obs.grid) > 0
    assert obs.current_step == 0
    assert obs.done is False


@pytest.mark.parametrize("task_id", ["reach_goal", "collect_and_escape", "survive_and_escape"])
def test_step_increments(task_id):
    env = GridWorldEnv(task_id, seed=1)
    env.reset()
    _, reward, done, info = env.step("RIGHT")
    assert info["step"] == 1
    assert 0.0 < reward.score < 1.0


@pytest.mark.parametrize("task_id", ["reach_goal", "collect_and_escape", "survive_and_escape"])
def test_max_steps_ends_episode(task_id):
    env = GridWorldEnv(task_id, seed=1)
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done, info = env.step("STAY")
        steps += 1
        if steps > 200:
            break
    assert done


def test_invalid_action_defaults_to_stay():
    env = GridWorldEnv("reach_goal", seed=1)
    env.reset()
    _, reward, _, _ = env.step("JUMP")
    assert 0.0 < reward.score < 1.0


def test_state_returns_grid():
    env = GridWorldEnv("reach_goal", seed=1)
    env.reset()
    state = env.state()
    assert state.agent_pos is not None
    assert state.goal_pos is not None


def test_reward_in_bounds():
    env = GridWorldEnv("survive_and_escape", seed=42)
    env.reset()
    for action in ["RIGHT","DOWN","RIGHT","DOWN","LEFT","UP"]:
        _, reward, done, _ = env.step(action)
        assert 0.0 < reward.score < 1.0
        if done:
            break


def test_deterministic_with_seed():
    def run(seed):
        env = GridWorldEnv("reach_goal", seed=seed)
        env.reset()
        for a in ["RIGHT","DOWN","RIGHT","DOWN"]:
            _, r, done, _ = env.step(a)
            if done:
                return r.score
        return r.score
    assert run(7) == run(7)


def test_step_after_done_raises():
    env = GridWorldEnv("reach_goal", seed=1)
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step("STAY")
        steps += 1
        if steps > 200: break
    with pytest.raises(RuntimeError):
        env.step("UP")


def test_invalid_task_raises():
    with pytest.raises(ValueError):
        GridWorldEnv("nonexistent_task")
