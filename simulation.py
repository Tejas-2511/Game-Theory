"""
Multi-Round Simulation Engine for Social Media Content Competition
Simulates repeated rounds of competition between adaptive creator strategies.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from game_theory import GameModel


class BaseCreatorAgent:
    def __init__(self, name: str, strategies: List[str]):
        self.name = name
        self.strategies = strategies
        self.history_own: List[int] = []
        self.history_opponent: List[int] = []
        self.history_payoff: List[float] = []

    def reset(self):
        self.history_own.clear()
        self.history_opponent.clear()
        self.history_payoff.clear()

    def select_action(self, round_num: int, game: GameModel, is_player_a: bool) -> int:
        raise NotImplementedError

    def update(self, own_action: int, opponent_action: int, payoff: float):
        self.history_own.append(own_action)
        self.history_opponent.append(opponent_action)
        self.history_payoff.append(payoff)


class AlwaysStrategyAgent(BaseCreatorAgent):
    def __init__(self, name: str, strategies: List[str], target_strategy_idx: int):
        super().__init__(name, strategies)
        self.target_idx = target_strategy_idx

    def select_action(self, round_num: int, game: GameModel, is_player_a: bool) -> int:
        return self.target_idx


class TitForTatAgent(BaseCreatorAgent):
    """
    Starts with cooperation (Original), then replicates the opponent's previous choice.
    """
    def __init__(self, name: str, strategies: List[str], initial_choice: int = 1):
        super().__init__(name, strategies)
        self.initial_choice = initial_choice

    def select_action(self, round_num: int, game: GameModel, is_player_a: bool) -> int:
        if round_num == 0 or len(self.history_opponent) == 0:
            return min(self.initial_choice, len(self.strategies) - 1)
        # Mirror opponent's last move if within bounds
        last_opp = self.history_opponent[-1]
        return last_opp if last_opp < len(self.strategies) else 0


class BestResponseLearnerAgent(BaseCreatorAgent):
    """
    Myopic Best-Response: assumes the opponent will repeat their most recent action.
    """
    def __init__(self, name: str, strategies: List[str]):
        super().__init__(name, strategies)

    def select_action(self, round_num: int, game: GameModel, is_player_a: bool) -> int:
        if round_num == 0 or len(self.history_opponent) == 0:
            return 0  # Default to Trending initially

        last_opp = self.history_opponent[-1]
        if is_player_a:
            best_responses = game.get_best_responses_a().get(last_opp, [0])
        else:
            best_responses = game.get_best_responses_b().get(last_opp, [0])
        return np.random.choice(best_responses)


class FictitiousPlayAgent(BaseCreatorAgent):
    """
    Fictitious Play: keeps count of all previous opponent choices,
    forms empirical probability distribution, and plays the expected-payoff maximizing action.
    """
    def __init__(self, name: str, strategies: List[str]):
        super().__init__(name, strategies)

    def select_action(self, round_num: int, game: GameModel, is_player_a: bool) -> int:
        if round_num == 0 or len(self.history_opponent) == 0:
            return 0

        counts = np.zeros(len(self.strategies))
        for opp_act in self.history_opponent:
            if opp_act < len(counts):
                counts[opp_act] += 1
        probs = counts / np.sum(counts)

        # Compute expected payoff for each candidate action
        expected_payoffs = np.zeros(len(self.strategies))
        for action_idx in range(len(self.strategies)):
            if is_player_a:
                expected_payoffs[action_idx] = np.dot(game.payoff_a[action_idx, :], probs)
            else:
                expected_payoffs[action_idx] = np.dot(game.payoff_b[:, action_idx], probs)

        best_actions = np.where(np.isclose(expected_payoffs, np.max(expected_payoffs)))[0]
        return np.random.choice(best_actions)


class EpsilonGreedyQLearner(BaseCreatorAgent):
    """
    Reinforcement learning agent maintaining Q-values for each strategy.
    """
    def __init__(self, name: str, strategies: List[str], epsilon: float = 0.15, alpha: float = 0.25):
        super().__init__(name, strategies)
        self.epsilon = epsilon
        self.alpha = alpha
        self.q_table = np.zeros(len(strategies), dtype=float)

    def reset(self):
        super().reset()
        self.q_table = np.zeros(len(self.strategies), dtype=float)

    def select_action(self, round_num: int, game: GameModel, is_player_a: bool) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.strategies))
        max_q = np.max(self.q_table)
        best_acts = np.where(np.isclose(self.q_table, max_q))[0]
        return np.random.choice(best_acts)

    def update(self, own_action: int, opponent_action: int, payoff: float):
        super().update(own_action, opponent_action, payoff)
        # Update Q-value via TD error
        self.q_table[own_action] += self.alpha * (payoff - self.q_table[own_action])


class RandomAgent(BaseCreatorAgent):
    def __init__(self, name: str, strategies: List[str], weights: List[float] = None):
        super().__init__(name, strategies)
        if weights is None:
            self.weights = np.ones(len(strategies)) / len(strategies)
        else:
            w = np.array(weights, dtype=float)
            self.weights = w / np.sum(w)

    def select_action(self, round_num: int, game: GameModel, is_player_a: bool) -> int:
        return np.random.choice(len(self.strategies), p=self.weights)


def get_agent_by_type(agent_type: str, name: str, strategies: List[str]) -> BaseCreatorAgent:
    if agent_type == "Always Trending (Viral Chaser)":
        return AlwaysStrategyAgent(name, strategies, target_strategy_idx=0)
    elif agent_type == "Always Original (Quality Purist)":
        return AlwaysStrategyAgent(name, strategies, target_strategy_idx=1)
    elif agent_type == "Always Niche (Community Specialist)" and len(strategies) > 2:
        return AlwaysStrategyAgent(name, strategies, target_strategy_idx=2)
    elif agent_type == "Tit-for-Tat (Reciprocal Cooperator)":
        return TitForTatAgent(name, strategies, initial_choice=1)
    elif agent_type == "Best-Response Learner (Greedy Adaptor)":
        return BestResponseLearnerAgent(name, strategies)
    elif agent_type == "Fictitious Play (Belief-Based Strategy)":
        return FictitiousPlayAgent(name, strategies)
    elif agent_type == "Q-Learning AI (Reinforcement Agent)":
        return EpsilonGreedyQLearner(name, strategies)
    else:
        return RandomAgent(name, strategies)


class MultiRoundSimulator:
    def __init__(self, game: GameModel, agent_a: BaseCreatorAgent, agent_b: BaseCreatorAgent):
        self.game = game
        self.agent_a = agent_a
        self.agent_b = agent_b

    def run_simulation(
        self,
        num_rounds: int = 30,
        noise_std: float = 0.0,
        trend_fatigue_decay: float = 0.0,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Executes num_rounds of competition.
        noise_std: adds stochastic volatility to payoffs (algorithm luck).
        trend_fatigue_decay: penalty if Trending is chosen repeatedly by both creators.
        """
        np.random.seed(seed)
        self.agent_a.reset()
        self.agent_b.reset()

        round_data = []
        cum_a = 0.0
        cum_b = 0.0
        consecutive_trending = 0

        nash_cells = set(self.game.find_pure_nash_equilibria())
        social_opt_cells = {(c[0], c[1]) for c in self.game.find_social_optimum()}

        for r in range(num_rounds):
            action_a = self.agent_a.select_action(r, self.game, is_player_a=True)
            action_b = self.agent_b.select_action(r, self.game, is_player_a=False)

            base_payoff_a = self.game.payoff_a[action_a, action_b]
            base_payoff_b = self.game.payoff_b[action_a, action_b]

            # Trending fatigue dynamics
            is_both_trending = (action_a == 0 and action_b == 0)
            if is_both_trending:
                consecutive_trending += 1
                penalty = 1.0 - (trend_fatigue_decay * min(consecutive_trending - 1, 5))
                penalty = max(0.4, penalty)
            else:
                consecutive_trending = 0
                penalty = 1.0

            # Algorithm virality stochastic shock
            noise_a = np.random.normal(0, noise_std) if noise_std > 0 else 0.0
            noise_b = np.random.normal(0, noise_std) if noise_std > 0 else 0.0

            actual_a = max(0.0, (base_payoff_a * penalty) + noise_a)
            actual_b = max(0.0, (base_payoff_b * penalty) + noise_b)

            cum_a += actual_a
            cum_b += actual_b

            self.agent_a.update(action_a, action_b, actual_a)
            self.agent_b.update(action_b, action_a, actual_b)

            is_nash = (action_a, action_b) in nash_cells
            is_social = (action_a, action_b) in social_opt_cells

            round_data.append({
                "round": r + 1,
                "strategy_a": self.game.strategies[action_a],
                "strategy_b": self.game.strategies[action_b],
                "action_a_idx": action_a,
                "action_b_idx": action_b,
                "payoff_a": round(actual_a, 2),
                "payoff_b": round(actual_b, 2),
                "cum_payoff_a": round(cum_a, 2),
                "cum_payoff_b": round(cum_b, 2),
                "total_round_engagement": round(actual_a + actual_b, 2),
                "is_nash": is_nash,
                "is_social_optimum": is_social
            })

        # Aggregated performance stats
        total_a = cum_a
        total_b = cum_b
        nash_alignment_pct = (sum(1 for d in round_data if d["is_nash"]) / num_rounds) * 100
        social_alignment_pct = (sum(1 for d in round_data if d["is_social_optimum"]) / num_rounds) * 100

        return {
            "rounds": round_data,
            "total_payoff_a": round(total_a, 2),
            "total_payoff_b": round(total_b, 2),
            "avg_payoff_a": round(total_a / num_rounds, 2),
            "avg_payoff_b": round(total_b / num_rounds, 2),
            "combined_total": round(total_a + total_b, 2),
            "nash_alignment_pct": round(nash_alignment_pct, 1),
            "social_alignment_pct": round(social_alignment_pct, 1),
            "winner": "Creator A" if total_a > total_b else ("Creator B" if total_b > total_a else "Tie")
        }
