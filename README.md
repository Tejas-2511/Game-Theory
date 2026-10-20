# 📱 Game Theory-Based Analysis of Social Media Content Competition

An interactive computational simulation examining content creator incentives, viral algorithms, Nash equilibria, and collective attention dilemmas in short-form video platforms (TikTok, Instagram Reels, YouTube Shorts).

Developed for academic evaluation combining Game Theory concepts, Python simulations, numerical results, and interactive visual dashboards.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure you have Python 3.9+ installed with required libraries:
```bash
pip install streamlit numpy pandas altair matplotlib seaborn
```

### 2. Launch the Application
Run the Streamlit app from the workspace folder:
```bash
streamlit run app.py
```
Or on Windows, simply double-click:
```bat
run.bat
```
The interactive application will automatically open in your default browser at `http://localhost:8501`.

---

## 🎯 Project Overview & Motivation

In algorithmic social platforms, competing creators vie for finite viewer attention and algorithm distribution. Each creator faces a strategic trade-off:
1. **Trending Content ($T$):** Mimic viral formats and audio. Low barrier to entry and high short-term virality, but creates feed congestion and viewer fatigue when duplicated.
2. **Original Content ($O$):** Invest in unique, high-effort production. Builds loyal audience engagement and retention, but risks being overshadowed by viral spikes.
3. **Niche Content ($N$):** Focus on dedicated subcultures. Insulates the creator from mainstream feed cannibalization.

Game Theory allows us to formally model these strategic interactions, identify stable outcomes (Nash Equilibria), and quantify the systemic inefficiencies of attention competition.

---

## 🧠 The Game Theory Model

### 1. The Normal-Form Strategic Payoff Matrix (2×2)

| Creator A \ Creator B | Trending ($T$) | Original ($O$) |
|:---|:---:|:---:|
| **Trending ($T$)** | **(3, 3)** <br> *(Nash Equilibrium)* | **(6, 2)** <br> *(Viral Steal)* |
| **Original ($O$)** | **(2, 6)** <br> *(Viral Steal)* | **(5, 5)** <br> *(Social Optimum)* |

*(Payoff values represent standardized engagement units: views, watch time, shares, and retention).*

### 2. Strategic Insights & Proofs

- **Dominant Strategy:**
  - For Creator A:
    - If B plays Trending: $u_A(T, T) = 3 > u_A(O, T) = 2$ $\rightarrow$ Trending is optimal.
    - If B plays Original: $u_A(T, O) = 6 > u_A(O, O) = 5$ $\rightarrow$ Trending is optimal.
  - Trending is a **strictly dominant strategy** for both creators.
- **Pure-Strategy Nash Equilibrium:**
  - Unique Nash Equilibrium at **$(Trending, Trending)$ with payoffs $(3, 3)$**.
  - At this point, neither creator can unilaterally improve their payoff by switching to Original.
- **The Attention Economy Prisoner's Dilemma:**
  - While $(Trending, Trending)$ is the rational equilibrium, both creators would be strictly better off at **$(Original, Original) = (5, 5)$**.
  - Selfish optimization leads to collective loss of engagement.
- **Price of Anarchy (PoA):**
  $$\text{PoA} = \frac{\text{Maximum Social Welfare}}{\text{Nash Equilibrium Welfare}} = \frac{5 + 5}{3 + 3} = \frac{10}{6} \approx 1.67$$
  Uncoordinated individual competition leads to an approximate **40% destruction of total potential attention value**.

---

## 💻 Application Features

The interactive dashboard includes 5 comprehensive modules:

1. **📊 Strategic Matrix & Equilibrium Solver**:
   - Interactive normal-form payoff matrix with visual indicator badges for Best Responses (BR A / BR B), Nash Equilibria, and Social Optima.
   - Mixed-strategy Nash equilibrium derivation with indifference formulas.
   - Scenario presets (Prisoner's Dilemma, Stag Hunt, Hawk-Dove) plus full 3×3 matrix editor.
2. **🎯 Single-Round What-If Sandbox**:
   - Select strategies for both creators in real-time.
   - Immediate feedback showing whether an outcome is stable or which creator has an incentive to defect.
   - Comparative engagement bar charts.
3. **🔄 Dynamic Iterated Multi-Round Simulation**:
   - Run 10 to 100 consecutive content publishing cycles.
   - Built-in behavioral agents: *Always Trending*, *Always Original*, *Tit-for-Tat*, *Best-Response Learner*, *Fictitious Play*, *Q-Learning AI*, and *Random Agent*.
   - Platform noise parameter ($\sigma$) modeling recommendation algorithm randomness.
   - Audience trend fatigue decay factor penalizing consecutive viral spam.
   - Interactive cumulative engagement trajectories and strategy adoption charts.
   - Downloadable CSV log of tournament results.
4. **🎮 Human vs. AI Creator Arena**:
   - Play as Creator A round-by-round against an adaptive AI competitor.
   - Real-time coaching tips explaining game-theoretic dynamics after every post.
5. **📑 Academic Report & Viva Cheat-Sheet**:
   - Structured academic summary covering problem statement, mathematical proofs, real-world social media implications, and answers to common examiner questions.

---

## 📂 Project Architecture

```
Game Theory/
│
├── app.py              # Main interactive Streamlit application with modern UI
├── game_theory.py      # Core game theory mathematical engine (Nash, BR, PoA)
├── simulation.py       # Iterated multi-round simulator & agent behavioral policies
├── run.bat             # One-click Windows launcher
└── README.md           # Complete project documentation & theory guide
```

---

## 🎓 Viva Q&A Highlights

- **Q: Why is (Trending, Trending) a Nash equilibrium if (Original, Original) gives higher payoffs?**
  - **A:** In a Nash equilibrium, no player can unilaterally deviate to improve their score. At $(Original, Original) = (5, 5)$, if Creator A unilaterally switches to *Trending*, their score jumps from $5$ to $6$. Both creators have this private incentive, drawing them into the $(3, 3)$ defect state.
- **Q: How does the iterated game resolve this dilemma?**
  - **A:** According to the **Folk Theorem**, in repeated games where creators care about future payoffs, reciprocal strategies like **Tit-for-Tat** can sustain mutual cooperation $(Original, Original)$ through the credible threat of retaliatory trend spamming.
- **Q: What role does the platform algorithm play?**
  - **A:** The recommendation algorithm defines the payoff matrix. If platforms implement fatigue penalties for redundant trend copies, they shift the incentives so that high-quality original content becomes the dominant strategy.
