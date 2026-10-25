# 📱 Game Theory: Social Media Content Competition
### Theoretical Foundations, Mathematical Proofs & Presentation Guide

> **Academic Evaluation & Viva Reference Document**  
> Formally modeling creator incentives, algorithmic payoffs, Nash equilibria, Pareto efficiency, and the Folk Theorem in short-form video ecosystems (TikTok, Instagram Reels, YouTube Shorts).

---

## 📑 Table of Contents
1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [Formal Game-Theoretic Modeling](#2-formal-game-theoretic-modeling)
3. [Core Concepts & Mathematical Proofs](#3-core-concepts--mathematical-proofs)
   - [3.1 Best-Response Correspondence](#31-best-response-correspondence)
   - [3.2 Dominant Strategy Analysis](#32-dominant-strategy-analysis)
   - [3.3 Pure-Strategy Nash Equilibrium](#33-pure-strategy-nash-equilibrium)
   - [3.4 Mixed-Strategy Nash Equilibrium](#34-mixed-strategy-nash-equilibrium)
   - [3.5 Pareto Optimality vs. Social Optimum](#35-pareto-optimality-vs-social-optimum)
   - [3.6 Price of Anarchy (PoA) & Welfare Inefficiency](#36-price-of-anarchy-poa--welfare-inefficiency)
4. [Game Typology & Scenario Analysis](#4-game-typology--scenario-analysis)
   - [Scenario A: Prisoner's Dilemma (The Viral Rat Race)](#scenario-a-prisoners-dilemma-the-viral-rat-race)
   - [Scenario B: Stag Hunt (Quality Coordination)](#scenario-b-stag-hunt-quality-coordination)
   - [Scenario C: Hawk-Dove / Chicken (Audience Congestion)](#scenario-c-hawk-dove--chicken-audience-congestion)
   - [Scenario D: 3×3 Extended Model (Niche Content Buffer)](#scenario-d-33-extended-model-niche-content-buffer)
5. [Repeated Game Extension: The Folk Theorem](#5-repeated-game-extension-the-folk-theorem)
   - [Finite Horizon & Backward Induction](#51-finite-horizon--backward-induction)
   - [Indefinite Horizon & The Shadow of the Future ($\delta$)](#52-indefinite-horizon--the-shadow-of-the-future-delta)
   - [Mathematical Derivation of the Critical Discount Factor ($\delta^*$)](#53-mathematical-derivation-of-the-critical-discount-factor-delta)
   - [Axelrod's Reciprocity Principles (Tit-for-Tat)](#54-axelrods-reciprocity-principles-tit-for-tat)
6. [Platform Policy & Mechanism Design](#6-platform-policy--mechanism-design)
7. [Viva & Examiner Q&A Cheat-Sheet](#7-viva--examiner-qa-cheat-sheet)

---

## 1. Executive Summary & Problem Statement

In modern algorithmic social platforms, creators compete for finite viewer attention and recommendation distribution. Each creator faces a strategic trade-off:
1. **Trending Content ($T$):** Mimicking viral audio clips, memes, and trending formats. Requires low production effort and yields rapid distribution, but saturates feeds and fatigues viewers when replicated.
2. **Original Content ($O$):** Producing unique, high-production-value, thoughtful material. Demands significant effort and delivers high audience retention, but risks getting drowned out by viral trend waves.
3. **Niche Content ($N$):** Catering exclusively to dedicated subcultures. Insulates creators from mainstream algorithmic volatility at the expense of mass reach.

**The Central Paradox:**  
While both creators and viewers benefit most when creators invest in high-effort, original content, individual algorithmic incentives drive creators into an exhaustive viral rat race—producing a classic **Prisoner's Dilemma**.

---

## 2. Formal Game-Theoretic Modeling

We formalize content competition as a normal-form strategic game:

$$\Gamma = \langle \mathcal{N}, (\mathcal{S}_i)_{i \in \mathcal{N}}, (u_i)_{i \in \mathcal{N}} \rangle$$

Where:
- **Players:** $\mathcal{N} = \{ \text{Creator A}, \text{Creator B} \}$
- **Strategy Sets:**  
  - For $2 \times 2$ Game: $\mathcal{S}_A = \mathcal{S}_B = \{ \text{Trending } (T), \text{Original } (O) \}$
  - For $3 \times 3$ Game: $\mathcal{S}_A = \mathcal{S}_B = \{ \text{Trending } (T), \text{Original } (O), \text{Niche } (N) \}$
- **Payoff Functions:** $u_i(s_A, s_B) \in \mathbb{R}$ quantifies total creator utility:
  $$u_i = \alpha \cdot \text{Reach} + \beta \cdot \text{Watch Time} + \gamma \cdot \text{Audience Loyalty} - c(\text{Production Cost})$$

### The Canonical Payoff Matrix ($2 \times 2$)

| Creator A \ Creator B | Trending ($T$) | Original ($O$) |
| :--- | :---: | :---: |
| **Trending ($T$)** | $(3.0, 3.0)$ <br> *(Nash Equilibrium)* | $(6.0, 2.0)$ <br> *(Viral Steal)* |
| **Original ($O$)** | $(2.0, 6.0)$ <br> *(Viral Steal)* | $(5.0, 5.0)$ <br> *(Social Optimum)* |

*Payoff values are calibrated standardized engagement units.*

---

## 3. Core Concepts & Mathematical Proofs

### 3.1 Best-Response Correspondence
The best-response correspondence identifies the payoff-maximizing strategy given an opponent's action:

$$BR_i(s_{-i}) = \arg\max_{s_i \in \mathcal{S}_i} u_i(s_i, s_{-i})$$

- **For Creator A (Row Player):**
  - Given B plays $T$: $u_A(T, T) = 3.0 > u_A(O, T) = 2.0 \implies BR_A(T) = T$
  - Given B plays $O$: $u_A(T, O) = 6.0 > u_A(O, O) = 5.0 \implies BR_A(O) = T$
- **For Creator B (Column Player):**
  - Given A plays $T$: $u_B(T, T) = 3.0 > u_B(T, O) = 2.0 \implies BR_B(T) = T$
  - Given A plays $O$: $u_B(O, T) = 6.0 > u_B(O, O) = 5.0 \implies BR_B(O) = T$

---

### 3.2 Dominant Strategy Analysis
A strategy $s_i^*$ is **strictly dominant** if it yields a strictly higher payoff than any other strategy, regardless of what the opponent plays:

$$\forall s_i \neq s_i^*, \quad \forall s_{-i} \in \mathcal{S}_{-i}: \quad u_i(s_i^*, s_{-i}) > u_i(s_i, s_{-i})$$

**Proof for Creator A:**
- State 1 ($B = T$): $u_A(T, T) - u_A(O, T) = 3.0 - 2.0 = +1.0 > 0$
- State 2 ($B = O$): $u_A(T, O) - u_A(O, O) = 6.0 - 5.0 = +1.0 > 0$

Since $u_A(T, s_B) > u_A(O, s_B)$ for all $s_B \in \{T, O\}$, **Trending ($T$) is a strictly dominant strategy** for Creator A.  
By matrix symmetry, **Trending ($T$) is strictly dominant** for Creator B.

> **Presentation Takeaway:** A rational creator never needs to guess or predict the rival's move; *Trending* is individually optimal under every circumstance.

---

### 3.3 Pure-Strategy Nash Equilibrium
A strategy profile $s^* = (s_A^*, s_B^*)$ is a **Nash Equilibrium** if no player can unilaterally deviate to improve their payoff:

$$\forall i \in \mathcal{N}, \quad \forall s_i \in \mathcal{S}_i: \quad u_i(s_i^*, s_{-i}^*) \ge u_i(s_i, s_{-i}^*)$$

Equivalently, $s_A^* \in BR_A(s_B^*)$ and $s_B^* \in BR_B(s_A^*)$.

- $BR_A(T) = \{T\}$ and $BR_B(T) = \{T\}$
- The intersection of best responses occurs uniquely at:
  $$s^* = (T, T) \quad \text{with payoffs } (3.0, 3.0)$$

**Stability Proof (Unilateral Deviation Check):**
- If Creator A unilaterally switches to $O$: Payoff drops from $3.0 \to 2.0$ ($\Delta = -1.0$).
- If Creator B unilaterally switches to $O$: Payoff drops from $3.0 \to 2.0$ ($\Delta = -1.0$).
Neither player has an incentive to deviate. $(T, T)$ is a **strict, stable Nash Equilibrium**.

---

### 3.4 Mixed-Strategy Nash Equilibrium
In a mixed strategy, players randomize over pure strategies:
- Creator A plays $T$ with probability $p$, and $O$ with probability $1 - p$.
- Creator B plays $T$ with probability $q$, and $O$ with probability $1 - q$.

#### Indifference Principle
In a mixed equilibrium, each player must randomize such that the opponent is **indifferent** between their pure strategies:

$$\mathbb{E}[u_A(T, q)] = \mathbb{E}[u_A(O, q)]$$

$$q \cdot u_A(T, T) + (1 - q) \cdot u_A(T, O) = q \cdot u_A(O, T) + (1 - q) \cdot u_A(O, O)$$

Substituting values:
$$q(3) + (1 - q)(6) = q(2) + (1 - q)(5)$$
$$6 - 3q = 5 - 3q \implies 6 = 5 \quad (\text{Contradiction!})$$

$$\text{Denominator: } (A_{00} - A_{01} - A_{10} + A_{11}) = (3 - 6 - 2 + 5) = 0$$

> **Theoretical Insight:** When a player possesses a strictly dominant pure strategy, the expected payoff slopes never intersect inside $q \in (0, 1)$. Therefore, **no interior mixed-strategy Nash equilibrium exists** in the Prisoner's Dilemma.

*(Note: In non-dominant games like Hawk-Dove or Stag Hunt, the indifference condition yields a valid interior mixed equilibrium).*

---

### 3.5 Pareto Optimality vs. Social Optimum

#### Definitions
- **Social Optimum:** The strategy profile that maximizes the sum of all utilities (Utilitarian Social Welfare):
  $$W(s) = \sum_{i \in \mathcal{N}} u_i(s)$$
  - $W(T, T) = 3.0 + 3.0 = 6.0$
  - $W(T, O) = 6.0 + 2.0 = 8.0$
  - $W(O, T) = 2.0 + 6.0 = 8.0$
  - $W(O, O) = 5.0 + 5.0 = \mathbf{10.0} \quad \implies \text{Social Optimum} = (O, O)$

- **Pareto Dominance:** Strategy profile $s$ Pareto-dominates profile $s'$ if:
  $$\forall i \in \mathcal{N}, \quad u_i(s) \ge u_i(s') \quad \text{and} \quad \exists j \in \mathcal{N}: u_j(s) > u_j(s')$$

- **Pareto Efficiency:** A profile $s$ is Pareto optimal (efficient) if no other profile Pareto-dominates it.

**Comparison in the Content Economy:**
$$(O, O) = (5.0, 5.0) \quad \text{strictly Pareto-dominates} \quad (T, T) = (3.0, 3.0)$$
Both creators are strictly better off at $(O, O)$ (+2.0 points each).  
However, $(O, O)$ is **not a Nash equilibrium** because either player faces an individual incentive to exploit the other by defecting to Trending ($5.0 \to 6.0$).

---

### 3.6 Price of Anarchy (PoA) & Welfare Inefficiency
The **Price of Anarchy (PoA)** measures the systemic inefficiency caused by selfish, uncoordinated behavior:

$$\text{PoA} = \frac{\max_{s \in \mathcal{S}} W(s)}{\min_{s^* \in \mathcal{NE}} W(s^*)}$$

$$\text{PoA} = \frac{W(O, O)}{W(T, T)} = \frac{5.0 + 5.0}{3.0 + 3.0} = \frac{10.0}{6.0} \approx \mathbf{1.67}$$

$$\text{Efficiency Loss} = W(O, O) - W(T, T) = 10.0 - 6.0 = \mathbf{4.0 \text{ units (40\% destruction of total value)}}$$

> **Presentation Metric:** Decentralized creator competition destroys **40% of the platform's potential attention value** because algorithmic incentives compel creators into feed congestion rather than quality.

---

## 4. Game Typology & Scenario Analysis

The project provides instant preset configurations to contrast different economic environments:

```
+-----------------------------------------------------------------------------------------+
|                                    GAME TYPOLOGY SUMMARY                                |
+-----------------------+--------------------+-----------------------+--------------------+
| Scenario Preset       | Dominant Strategy? | Nash Equilibria       | Economic Dynamic   |
+-----------------------+--------------------+-----------------------+--------------------+
| Prisoner's Dilemma    | Yes (Trending)     | 1 Pure: (T, T)        | Viral Rat Race     |
| Stag Hunt             | No                 | 2 Pure: (O, O), (T, T)| Quality Trust      |
| Hawk-Dove / Chicken   | No                 | 2 Pure: (T, O), (O, T)| Feed Congestion    |
| 3x3 Extended          | No                 | 2 Pure: (T, N), (N, T)| Niche Buffering    |
+-----------------------+--------------------+-----------------------+--------------------+
```

### Scenario A: Prisoner's Dilemma (The Viral Rat Race)
- **Payoffs:** $A = \begin{pmatrix} 3 & 6 \\ 2 & 5 \end{pmatrix}, B = \begin{pmatrix} 3 & 2 \\ 6 & 5 \end{pmatrix}$
- **Equilibrium:** $(T, T) = (3.0, 3.0)$
- **Real-World Meaning:** Platforms heavily promote trending audio clips, guaranteeing individual short-term gains at the expense of audience fatigue.

### Scenario B: Stag Hunt (Quality Coordination)
- **Payoffs:** $A = \begin{pmatrix} 2 & 2 \\ 0 & 5 \end{pmatrix}, B = \begin{pmatrix} 2 & 0 \\ 2 & 5 \end{pmatrix}$
- **Equilibria:** Two pure Nash equilibria:
  1. $(O, O) = (5.0, 5.0)$ — **Payoff Dominant** (best combined reward).
  2. $(T, T) = (2.0, 2.0)$ — **Risk Dominant** (safe against rival defection).
- **Real-World Meaning:** Original content requires community trust. If both commit, engagement explodes; if one reneges, the original creator receives zero distribution.

### Scenario C: Hawk-Dove / Chicken (Audience Congestion)
- **Payoffs:** $A = \begin{pmatrix} 1 & 6 \\ 2 & 4 \end{pmatrix}, B = \begin{pmatrix} 1 & 2 \\ 6 & 4 \end{pmatrix}$
- **Equilibria:** Two asymmetric pure equilibria: $(T, O) = (6, 2)$ and $(O, T) = (2, 6)$, plus an interior mixed equilibrium.
- **Real-World Meaning:** Feeds cannot tolerate two identical viral clones. When both play Hawk (*Trending*), catastrophic cannibalization occurs ($1.0, 1.0$). One creator exploits while the other retreats to original content.

### Scenario D: 3×3 Extended Model (Niche Content Buffer)
- **Strategies:** Trending ($T$), Original ($O$), Niche ($N$)
- **Payoffs ($A$):**
  $$\begin{pmatrix} 3 & 6 & 5 \\ 2 & 5 & 4 \\ 4 & 4 & 3 \end{pmatrix}$$
- **Equilibria:** $(T, N)$ and $(N, T)$
- **Real-World Meaning:** *Niche* content insulates creators. Even when a rival floods the mainstream feed with *Trending* content, a *Niche* creator retains a loyal subculture ($u = 4.0$), mitigating the harshness of the Prisoner's Dilemma.

---

## 5. Repeated Game Extension: The Folk Theorem (Concise Summary)

While chasing viral trends strictly dominates in a **single-shot game** (trapping creators at $3.0, 3.0$), real competition is ongoing. The **Folk Theorem** explains how repeated interaction can sustain mutual cooperation $(Original, Original) = (5.0, 5.0)$:

1. **Finite vs. Indefinite Horizon:**
   - **Known Fixed Rounds ($T$):** By backward induction, anticipating defection in round $T$ causes cooperation to unravel back to round 1.
   - **Indefinite Horizon ($\delta$):** When future rounds are expected with continuation probability / discount factor $\delta \in (0, 1)$, future payoffs matter.

2. **The Cooperation Condition:**
   Under a reciprocal trigger strategy (e.g. Tit-for-Tat), mutual cooperation is a sustainable Subgame Perfect Equilibrium if:
   $$\delta \ge \delta^* = \frac{u(\text{Temptation}) - u(\text{Cooperation})}{u(\text{Temptation}) - u(\text{Punishment})} = \frac{6.0 - 5.0}{6.0 - 3.0} \approx \mathbf{0.33}$$
   *Meaning: As long as creators value future audience retention at $\ge 33\%$ of immediate virality, the threat of retaliatory trend-chasing deters defection.*

3. **Tit-for-Tat Mechanism:** Starts with *Original*, punishes trend-jacking immediately, and forgives as soon as mutual quality resumes.

---

## 6. Platform Policy & Mechanism Design

How can platforms like YouTube or Instagram escape the Prisoner's Dilemma through algorithmic mechanism design?

1. **Algorithmic Trend Fatigue Penalties:**
   - Reducing distribution multipliers for redundant trend copies reduces $u(T, O)$ from $6.0 \to 4.5$.
   - This lowers the critical discount factor:
     $$\delta^* = \frac{4.5 - 5.0}{4.5 - 3.0} < 0 \implies \text{Original content becomes strictly dominant!}$$
2. **Subscriber Loyalty Monopolies:**
   - Shifting feed weighting from ephemeral global feeds (TikTok style) to subscriber feeds (YouTube subscription model) increases creator patience $\delta \to 1.0$.

---

## 7. Viva & Examiner Q&A Cheat-Sheet

#### Q1: Why do rational creators end up at $(3.0, 3.0)$ when $(5.0, 5.0)$ is clearly better for both?
> **Answer:** In game theory, rationality is individual, not collective. At $(5.0, 5.0)$, Creator A knows that switching to *Trending* increases their individual payoff to $6.0$. Because Creator B faces the exact same incentive, both unilaterally defect, pulling the system into the Nash equilibrium $(3.0, 3.0)$ where neither can improve by deviating alone.

#### Q2: Can a mixed-strategy Nash equilibrium resolve this dilemma?
> **Answer:** No. In a standard Prisoner's Dilemma, *Trending* strictly dominates *Original* for all opponent choices. The expected payoff equations never intersect within the valid probability interval $p, q \in (0, 1)$, precluding any interior mixed equilibrium.

#### Q3: What is the economic significance of the Price of Anarchy ($\approx 1.67$)?
> **Answer:** Price of Anarchy quantifies the efficiency penalty of decentralization. A PoA of $1.67$ ($10.0 / 6.0$) signifies that unregulated individual incentives destroy approximately $40\%$ of the maximum potential engagement value that could be achieved under coordinated cooperation.

#### Q4: Why does backward induction prevent cooperation in finite games?
> **Answer:** In a known final round $T$, there is no future to retaliate against defection. Thus, both players defect. Anticipating this, creators also defect in $T-1$, cascading back to round 1. Cooperation requires an indefinite horizon or uncertainty regarding the game's termination.

#### Q5: What does the threshold $\delta^* \approx 0.33$ represent?
> **Answer:** It represents the minimum patience required for future punishment to outweigh current temptation. If a creator values the next round at least $33\%$ as much as the current round, the long-term cost of being punished at $(3.0, 3.0)$ exceeds the one-round gain of defecting from $5.0 \to 6.0$.
