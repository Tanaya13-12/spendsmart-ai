"""
SpendSmart AI - Reinforcement Learning (Q-Learning) Budget Simulator
Implements:
- Practical 11: Tabular Q-Learning for dynamic monthly budget and savings allocation.
- Formulates State space, Action space, Reward function, and Epsilon-Greedy Exploration.
- Simulates adaptive budget management respecting essential expense constraints.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# State Space Definitions (3 x 3 x 3 = 27 discrete states)
BUDGET_STATUS_LABELS = ["Under_Budget", "On_Budget", "Over_Budget"]
SAVINGS_PROGRESS_LABELS = ["Behind", "On_Track", "Ahead"]
EXPENSE_PRESSURE_LABELS = ["Low", "Moderate", "High"]

# Action Space Definitions
ACTION_NAMES = [
    "Allocate Extra to Savings (+15%)",
    "Reduce Discretionary Spending (-20%)",
    "Maintain Current Allocation",
    "Reallocate Surplus to Essential Buffer",
]


class BudgetEnvironment:
    """
    Simulated personal finance environment with monthly income,
    essential expenses, discretionary spending pressures, and a savings goal.
    """

    def __init__(
        self,
        monthly_income: float = 5800.0,
        essential_expenses: float = 2300.0,  # Rent, groceries, utilities
        target_savings: float = 1200.0,
        random_seed: int = 42,
    ):
        self.monthly_income = monthly_income
        self.essential_expenses = essential_expenses
        self.target_savings = target_savings
        self.rng = np.random.default_rng(random_seed)

        # Baseline discretionary allowance
        self.discretionary_budget = max(
            0.0, self.monthly_income - self.essential_expenses - self.target_savings
        )

        self.reset()

    def reset(self) -> int:
        """Resets the environment to an initial state at the beginning of an episode."""
        self.current_savings = 0.0
        self.current_discretionary = self.discretionary_budget
        self.cumulative_savings = 0.0
        self.month = 1

        # Initial random state
        self.b_idx = 1  # On_Budget
        self.s_idx = 0  # Behind
        self.p_idx = int(self.rng.choice([0, 1, 2]))  # Low, Mod, or High pressure

        return self.get_state_index(self.b_idx, self.s_idx, self.p_idx)

    @staticmethod
    def get_state_index(b_idx: int, s_idx: int, p_idx: int) -> int:
        """Converts (budget_idx, savings_idx, pressure_idx) tuple to unique state integer [0..26]."""
        return b_idx * 9 + s_idx * 3 + p_idx

    @staticmethod
    def decode_state_index(state_idx: int) -> Tuple[str, str, str]:
        """Decodes state integer [0..26] to human-readable labels."""
        b_idx = state_idx // 9
        s_idx = (state_idx % 9) // 3
        p_idx = state_idx % 3
        return (
            BUDGET_STATUS_LABELS[b_idx],
            SAVINGS_PROGRESS_LABELS[s_idx],
            EXPENSE_PRESSURE_LABELS[p_idx],
        )

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """
        Executes an action in the environment:
        0: Allocate extra to savings
        1: Reduce discretionary
        2: Maintain
        3: Reallocate to essential buffer
        """
        # Exogenous monthly expense pressure shock
        pressure_factor = 0.85 if self.p_idx == 0 else (1.05 if self.p_idx == 1 else 1.35)
        # Random noise in actual incurred discretionary expenses
        actual_discretionary = self.current_discretionary * pressure_factor * self.rng.uniform(0.9, 1.1)

        # Apply Agent Action
        savings_allocated = self.target_savings
        essential_allocated = self.essential_expenses

        if action == 0:  # Allocate extra to savings
            extra_savings = min(300.0, max(0.0, self.monthly_income - essential_allocated - actual_discretionary))
            savings_allocated += extra_savings
            actual_discretionary = max(200.0, actual_discretionary - extra_savings)
        elif action == 1:  # Reduce discretionary spending
            actual_discretionary = max(200.0, actual_discretionary * 0.80)
            # Reinvest savings from reduction
            savings_allocated += (self.discretionary_budget - actual_discretionary) * 0.5
        elif action == 2:  # Maintain
            pass
        elif action == 3:  # Reallocate surplus to essential buffer
            buffer_amt = 150.0
            essential_allocated += buffer_amt
            actual_discretionary = max(200.0, actual_discretionary - buffer_amt)

        total_spent = essential_allocated + actual_discretionary
        surplus = self.monthly_income - total_spent - savings_allocated

        # Determine next state
        # 1. Budget status
        if total_spent < (self.monthly_income - self.target_savings * 0.9):
            next_b = 0  # Under_Budget
        elif total_spent <= self.monthly_income - self.target_savings * 0.5:
            next_b = 1  # On_Budget
        else:
            next_b = 2  # Over_Budget

        # 2. Savings progress
        self.cumulative_savings += savings_allocated
        monthly_expected = self.target_savings * self.month
        if self.cumulative_savings < monthly_expected * 0.9:
            next_s = 0  # Behind
        elif self.cumulative_savings <= monthly_expected * 1.1:
            next_s = 1  # On_Track
        else:
            next_s = 2  # Ahead

        # 3. Next month pressure shock
        next_p = int(self.rng.choice([0, 1, 2], p=[0.3, 0.5, 0.2]))

        # Calculate Reward
        reward = 0.0
        # Positive reward for staying under/on budget
        if next_b == 0:
            reward += 20.0
        elif next_b == 1:
            reward += 10.0
        else:
            reward -= 25.0  # Over budget penalty

        # Savings achievement reward
        if next_s == 2:
            reward += 20.0
        elif next_s == 1:
            reward += 10.0
        else:
            reward -= 15.0  # Behind on savings penalty

        # Deficit penalty (if spending + savings exceeded income)
        if surplus < 0:
            reward -= abs(surplus) * 0.05

        # Protect essentials (severe penalty if essentials are underfunded)
        if essential_allocated < self.essential_expenses:
            reward -= 50.0

        self.b_idx, self.s_idx, self.p_idx = next_b, next_s, next_p
        self.month += 1
        done = self.month > 12  # 1-year simulation per episode

        next_state = self.get_state_index(next_b, next_s, next_p)
        info = {
            "month": self.month - 1,
            "total_spent": total_spent,
            "savings_allocated": savings_allocated,
            "essential_allocated": essential_allocated,
            "discretionary_spent": actual_discretionary,
            "surplus": surplus,
        }

        return next_state, reward, done, info


class QLearningAgent:
    """
    Tabular Q-Learning Agent.
    Updates: Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
    Uses epsilon-greedy exploration with exponential or linear decay.
    """

    def __init__(
        self,
        n_states: int = 27,
        n_actions: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        exploration_rate: float = 1.0,
        exploration_decay: float = 0.995,
        min_exploration: float = 0.05,
        seed: int = 42,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration
        self.rng = np.random.default_rng(seed)

        # Initialize Q-table with zeros
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)

    def choose_action(self, state: int, greedy: bool = False) -> int:
        """Epsilon-greedy action selection."""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """Bellman equation Q-learning update."""
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward if done else reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error

    def decay_exploration(self):
        """Decay exploration rate epsilon after each episode."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


def train_q_learning_agent(
    episodes: int = 1000,
    learning_rate: float = 0.1,
    discount_factor: float = 0.95,
    exploration_rate: float = 1.0,
    exploration_decay: float = 0.995,
    monthly_income: float = 5800.0,
    essential_expenses: float = 2300.0,
    target_savings: float = 1200.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes full Q-Learning training loop for the budget simulator.
    Tracks episode rewards, epsilon decay, and extracts the optimal policy.
    """
    env = BudgetEnvironment(
        monthly_income=monthly_income,
        essential_expenses=essential_expenses,
        target_savings=target_savings,
        random_seed=seed,
    )

    agent = QLearningAgent(
        n_states=27,
        n_actions=4,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        exploration_rate=exploration_rate,
        exploration_decay=exploration_decay,
        seed=seed,
    )

    episode_rewards = []
    epsilon_history = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_ep_reward = 0.0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_ep_reward += reward

        agent.decay_exploration()
        episode_rewards.append(float(total_ep_reward))
        epsilon_history.append(float(agent.epsilon))

    # Rolling smoothed rewards (window = 30)
    smoothed_rewards = pd.Series(episode_rewards).rolling(window=30, min_periods=1).mean().tolist()

    # Formulate readable Q-Table and Optimal Policy DataFrame
    q_table_rows = []
    for s_idx in range(27):
        b_label, s_label, p_label = BudgetEnvironment.decode_state_index(s_idx)
        best_a = int(np.argmax(agent.q_table[s_idx]))
        q_table_rows.append({
            "State_ID": s_idx,
            "Budget_Status": b_label,
            "Savings_Progress": s_label,
            "Expense_Pressure": p_label,
            "Extra_Savings_Q": round(float(agent.q_table[s_idx, 0]), 2),
            "Reduce_Discretionary_Q": round(float(agent.q_table[s_idx, 1]), 2),
            "Maintain_Q": round(float(agent.q_table[s_idx, 2]), 2),
            "Buffer_Surplus_Q": round(float(agent.q_table[s_idx, 3]), 2),
            "Optimal_Action": ACTION_NAMES[best_a],
        })

    q_df = pd.DataFrame(q_table_rows)

    # Run 12-month simulation comparing Learned Agent vs Fixed Baseline
    baseline_env = BudgetEnvironment(
        monthly_income=monthly_income,
        essential_expenses=essential_expenses,
        target_savings=target_savings,
        random_seed=seed + 100,
    )
    agent_env = BudgetEnvironment(
        monthly_income=monthly_income,
        essential_expenses=essential_expenses,
        target_savings=target_savings,
        random_seed=seed + 100,
    )

    # Baseline always uses action 2 (Maintain current allocation rigidly)
    b_state = baseline_env.reset()
    baseline_history = []
    done = False
    while not done:
        _, r, done, info = baseline_env.step(action=2)
        baseline_history.append(info)

    # Trained Agent uses learned policy
    a_state = agent_env.reset()
    agent_history = []
    done = False
    while not done:
        opt_act = agent.choose_action(a_state, greedy=True)
        a_state, r, done, info = agent_env.step(action=opt_act)
        info["action_taken"] = ACTION_NAMES[opt_act]
        agent_history.append(info)

    sim_df_baseline = pd.DataFrame(baseline_history)
    sim_df_agent = pd.DataFrame(agent_history)

    return {
        "episodes": episodes,
        "learning_rate": learning_rate,
        "discount_factor": discount_factor,
        "final_epsilon": float(agent.epsilon),
        "episode_rewards": episode_rewards,
        "smoothed_rewards": smoothed_rewards,
        "epsilon_history": epsilon_history,
        "q_table": q_df.to_dict(orient="records"),
        "baseline_simulation": sim_df_baseline.to_dict(orient="records"),
        "agent_simulation": sim_df_agent.to_dict(orient="records"),
        "agent_cumulative_savings": float(sim_df_agent["savings_allocated"].sum()),
        "baseline_cumulative_savings": float(sim_df_baseline["savings_allocated"].sum()),
    }
