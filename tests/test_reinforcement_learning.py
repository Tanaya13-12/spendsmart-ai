"""
Automated unit tests for Practical 11: Reinforcement Learning (Q-Learning) budget simulator.
"""

import numpy as np
import pytest
from src.reinforcement_learning import (
    BudgetEnvironment,
    QLearningAgent,
    train_q_learning_agent,
)


def test_budget_environment_reset_and_state_encoding():
    """Verify state index stays within valid bound [0..26] and decodes cleanly."""
    env = BudgetEnvironment(monthly_income=5000.0, essential_expenses=2000.0, target_savings=1000.0, random_seed=42)
    s0 = env.reset()

    assert 0 <= s0 <= 26
    b_label, s_label, p_label = BudgetEnvironment.decode_state_index(s0)
    assert b_label in ["Under_Budget", "On_Budget", "Over_Budget"]
    assert s_label in ["Behind", "On_Track", "Ahead"]
    assert p_label in ["Low", "Moderate", "High"]


def test_budget_environment_step_transition():
    """Verify step transitions, financial calculations, and terminal flags."""
    env = BudgetEnvironment(monthly_income=6000.0, essential_expenses=2200.0, target_savings=1200.0, random_seed=42)
    env.reset()

    # Take action 1 (reduce discretionary)
    next_s, reward, done, info = env.step(action=1)

    assert 0 <= next_s <= 26
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "total_spent" in info
    assert "savings_allocated" in info
    # Protected essentials constraint: essential allocated must cover baseline
    assert info["essential_allocated"] >= env.essential_expenses


def test_q_learning_agent_bellman_update():
    """Verify tabular Q-value update shifts toward TD target."""
    agent = QLearningAgent(n_states=27, n_actions=4, learning_rate=0.5, discount_factor=0.9, exploration_rate=0.0)

    # Initial Q value is 0
    assert agent.q_table[0, 1] == 0.0

    # Apply transition: state 0, action 1, reward 10, next state 1
    # TD target = reward + gamma * max_a' Q(next_state, a') = 10 + 0 = 10
    # New Q = 0 + 0.5 * (10 - 0) = 5.0
    agent.update(state=0, action=1, reward=10.0, next_state=1, done=False)

    assert pytest.approx(agent.q_table[0, 1], rel=1e-3) == 5.0


def test_epsilon_decay():
    """Verify exploration rate ε decays toward min_epsilon."""
    agent = QLearningAgent(exploration_rate=1.0, exploration_decay=0.9, min_exploration=0.1)
    agent.decay_exploration()
    assert agent.epsilon == 0.9

    for _ in range(30):
        agent.decay_exploration()
    assert agent.epsilon >= 0.1


def test_full_q_learning_training_loop():
    """Verify complete training loop runs smoothly and produces valid output dictionary."""
    res = train_q_learning_agent(episodes=100, monthly_income=5000.0, essential_expenses=2000.0, target_savings=1000.0)

    assert res["episodes"] == 100
    assert len(res["episode_rewards"]) == 100
    assert len(res["smoothed_rewards"]) == 100
    assert len(res["q_table"]) == 27
    assert len(res["agent_simulation"]) == 12
    assert len(res["baseline_simulation"]) == 12
    assert res["agent_cumulative_savings"] > 0
