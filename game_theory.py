"""
Game Theory Engine for Social Media Content Competition
Handles payoff matrices, best responses, dominant strategies,
Nash equilibria (pure and mixed), and Pareto efficiency calculations.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np


class GameModel:
    def __init__(self, strategies: List[str], payoff_matrix_a: np.ndarray, payoff_matrix_b: np.ndarray):
        """
        payoff_matrix_a[i, j] = payoff to Creator A when A plays strategies[i] and B plays strategies[j]
        payoff_matrix_b[i, j] = payoff to Creator B when A plays strategies[i] and B plays strategies[j]
        """
        self.strategies = list(strategies)
        self.num_strategies = len(strategies)
        self.payoff_a = np.array(payoff_matrix_a, dtype=float)
        self.payoff_b = np.array(payoff_matrix_b, dtype=float)

    @classmethod
    def create_2x2_standard(cls, t_t=(3, 3), t_o=(6, 2), o_t=(2, 6), o_o=(5, 5)):
        """
        Creates standard 2-strategy game:
        Row / Col: [Trending, Original]
        """
        strategies = ["Trending", "Original"]
        p_a = np.array([
            [t_t[0], t_o[0]],
            [o_t[0], o_o[0]]
        ], dtype=float)
        p_b = np.array([
            [t_t[1], t_o[1]],
            [o_t[1], o_o[1]]
        ], dtype=float)
        return cls(strategies, p_a, p_b)

    @classmethod
    def create_3x3_extended(cls, matrix_a: Optional[np.ndarray] = None, matrix_b: Optional[np.ndarray] = None):
        """
        Creates 3-strategy game:
        Strategies: [Trending, Original, Niche]
        Default payoffs model realistic social media platform dynamics:
        - Trending vs Trending: split viral audience (3, 3)
        - Trending vs Original: trending steals mainstream feed (6, 2)
        - Trending vs Niche: trending gets viral hits, niche keeps hardcore loyalists (5, 4)
        - Original vs Trending: original drowned out in feeds (2, 6)
        - Original vs Original: collaborative high quality engagement (5, 5)
        - Original vs Niche: both thrive with differentiated audiences (4, 4)
        - Niche vs Trending: niche retains loyal subculture (4, 5)
        - Niche vs Original: niche retains modest distinct base (4, 4)
        - Niche vs Niche: niche competition divides dedicated followers (3, 3)
        """
        strategies = ["Trending", "Original", "Niche"]
        if matrix_a is None or matrix_b is None:
            p_a = np.array([
                [3.0, 6.0, 5.0],
                [2.0, 5.0, 4.0],
                [4.0, 4.0, 3.0]
            ])
            p_b = np.array([
                [3.0, 2.0, 4.0],
                [6.0, 5.0, 4.0],
                [5.0, 4.0, 3.0]
            ])
        else:
            p_a = np.array(matrix_a, dtype=float)
            p_b = np.array(matrix_b, dtype=float)
        return cls(strategies, p_a, p_b)

    def get_best_responses_a(self) -> Dict[int, List[int]]:
        """
        For each strategy j of Creator B (column), find the row(s) i maximizing Creator A's payoff.
        Returns: {col_idx: [best_row_idx, ...]}
        """
        best_a = {}
        for j in range(self.num_strategies):
            col_vals = self.payoff_a[:, j]
            max_val = np.max(col_vals)
            best_rows = np.where(np.isclose(col_vals, max_val))[0].tolist()
            best_a[j] = best_rows
        return best_a

    def get_best_responses_b(self) -> Dict[int, List[int]]:
        """
        For each strategy i of Creator A (row), find the column(s) j maximizing Creator B's payoff.
        Returns: {row_idx: [best_col_idx, ...]}
        """
        best_b = {}
        for i in range(self.num_strategies):
            row_vals = self.payoff_b[i, :]
            max_val = np.max(row_vals)
            best_cols = np.where(np.isclose(row_vals, max_val))[0].tolist()
            best_b[i] = best_cols
        return best_b

    def find_pure_nash_equilibria(self) -> List[Tuple[int, int]]:
        """
        Find (i, j) cell coordinates where:
        i is a best response to j for Creator A, AND
        j is a best response to i for Creator B.
        """
        best_a = self.get_best_responses_a()
        best_b = self.get_best_responses_b()

        nash_eqs = []
        for i in range(self.num_strategies):
            for j in range(self.num_strategies):
                if i in best_a[j] and j in best_b[i]:
                    nash_eqs.append((i, j))
        return nash_eqs

    def find_dominant_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Checks for strictly and weakly dominant strategies for both Creator A and B.
        """
        results = {"Creator A": {}, "Creator B": {}}

        # Check Creator A (rows)
        strict_dom_a = None
        weak_dom_a = None
        for i in range(self.num_strategies):
            is_strict = True
            is_weak = True
            for other_i in range(self.num_strategies):
                if i == other_i:
                    continue
                diff = self.payoff_a[i, :] - self.payoff_a[other_i, :]
                if not np.all(diff > 0):
                    is_strict = False
                if not (np.all(diff >= 0) and np.any(diff > 0)):
                    is_weak = False
            if is_strict:
                strict_dom_a = self.strategies[i]
            if is_weak and not is_strict:
                weak_dom_a = self.strategies[i]

        results["Creator A"]["strictly_dominant"] = strict_dom_a
        results["Creator A"]["weakly_dominant"] = weak_dom_a

        # Check Creator B (columns)
        strict_dom_b = None
        weak_dom_b = None
        for j in range(self.num_strategies):
            is_strict = True
            is_weak = True
            for other_j in range(self.num_strategies):
                if j == other_j:
                    continue
                diff = self.payoff_b[:, j] - self.payoff_b[:, other_j]
                if not np.all(diff > 0):
                    is_strict = False
                if not (np.all(diff >= 0) and np.any(diff > 0)):
                    is_weak = False
            if is_strict:
                strict_dom_b = self.strategies[j]
            if is_weak and not is_strict:
                weak_dom_b = self.strategies[j]

        results["Creator B"]["strictly_dominant"] = strict_dom_b
        results["Creator B"]["weakly_dominant"] = weak_dom_b

        return results

    def find_social_optimum(self) -> List[Tuple[int, int, float]]:
        """
        Find cell(s) that maximize combined total engagement (Creator A + Creator B).
        Returns: list of (i, j, total_payoff)
        """
        total = self.payoff_a + self.payoff_b
        max_total = np.max(total)
        cells = []
        for i in range(self.num_strategies):
            for j in range(self.num_strategies):
                if np.isclose(total[i, j], max_total):
                    cells.append((i, j, float(max_total)))
        return cells

    def find_pareto_efficient_outcomes(self) -> List[Tuple[int, int]]:
        """
        An outcome (i, j) is Pareto efficient if no other outcome (i', j')
        can make at least one creator strictly better off without making the other worse off.
        """
        outcomes = [(i, j) for i in range(self.num_strategies) for j in range(self.num_strategies)]
        pareto = []
        for (i, j) in outcomes:
            val_a, val_b = self.payoff_a[i, j], self.payoff_b[i, j]
            dominated = False
            for (oi, oj) in outcomes:
                if (oi, oj) == (i, j):
                    continue
                o_val_a, o_val_b = self.payoff_a[oi, oj], self.payoff_b[oi, oj]
                if (o_val_a >= val_a and o_val_b >= val_b) and (o_val_a > val_a or o_val_b > val_b):
                    dominated = True
                    break
            if not dominated:
                pareto.append((i, j))
        return pareto

    def calculate_mixed_nash_2x2(self) -> Optional[Dict[str, Any]]:
        """
        If this is a 2x2 game, computes the mixed strategy Nash equilibrium (p, q):
        p = prob(A plays Trending)
        q = prob(B plays Trending)
        """
        if self.num_strategies != 2:
            return None

        # For Creator A:
        # Expected payoff of A playing row 0 given B plays q: q*A00 + (1-q)*A01
        # Expected payoff of A playing row 1 given B plays q: q*A10 + (1-q)*A11
        # Equate payoffs to make A indifferent:
        # q*(A00 - A01 - A10 + A11) = A11 - A01
        a00, a01 = self.payoff_a[0, 0], self.payoff_a[0, 1]
        a10, a11 = self.payoff_a[1, 0], self.payoff_a[1, 1]

        b00, b01 = self.payoff_b[0, 0], self.payoff_b[0, 1]
        b10, b11 = self.payoff_b[1, 0], self.payoff_b[1, 1]

        denom_q = (a00 - a01 - a10 + a11)
        denom_p = (b00 - b10 - b01 + b11)

        if np.isclose(denom_q, 0) or np.isclose(denom_p, 0):
            return None

        q = (a11 - a01) / denom_q
        p = (b11 - b10) / denom_p

        is_valid = (0.0 <= p <= 1.0) and (0.0 <= q <= 1.0)
        return {
            "p_creator_a_trending": float(p),
            "p_creator_a_original": float(1.0 - p),
            "q_creator_b_trending": float(q),
            "q_creator_b_original": float(1.0 - q),
            "is_valid_internal": is_valid
        }

    def is_prisoners_dilemma(self) -> Dict[str, Any]:
        """
        Checks whether the 2x2 game fulfills Prisoner's Dilemma properties:
        1. Both have a strictly dominant strategy (e.g. Trending).
        2. The unique Nash equilibrium is Pareto suboptimal (both could do strictly better playing Original).
        """
        if self.num_strategies != 2:
            return {"is_pd": False, "reason": "Requires 2 strategies to classify standard Prisoner's Dilemma."}

        doms = self.find_dominant_strategies()
        dom_a = doms["Creator A"]["strictly_dominant"]
        dom_b = doms["Creator B"]["strictly_dominant"]
        nash_list = self.find_pure_nash_equilibria()
        social = self.find_social_optimum()

        if dom_a == "Trending" and dom_b == "Trending" and len(nash_list) == 1 and nash_list[0] == (0, 0):
            # Check if (Original, Original) gives strictly higher payoff to both
            nash_payoffs = (self.payoff_a[0, 0], self.payoff_b[0, 0])
            coop_payoffs = (self.payoff_a[1, 1], self.payoff_b[1, 1])
            if coop_payoffs[0] > nash_payoffs[0] and coop_payoffs[1] > nash_payoffs[1]:
                efficiency_loss = (coop_payoffs[0] + coop_payoffs[1]) - (nash_payoffs[0] + nash_payoffs[1])
                price_of_anarchy = (coop_payoffs[0] + coop_payoffs[1]) / (nash_payoffs[0] + nash_payoffs[1]) if sum(nash_payoffs) > 0 else 1.0
                return {
                    "is_pd": True,
                    "nash": (0, 0),
                    "cooperative": (1, 1),
                    "efficiency_loss": efficiency_loss,
                    "price_of_anarchy": round(price_of_anarchy, 2),
                    "reason": "Classic Social Media Prisoner's Dilemma detected! Both creators have an individual incentive to chase viral trends, leading to a suboptimal collective outcome."
                }

        return {
            "is_pd": False,
            "reason": "Does not strictly match canonical Prisoner's Dilemma conditions."
        }
