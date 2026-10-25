# 📱 Game Theory: Social Media Content Competition

An interactive computational simulation examining content creator incentives, viral algorithms, Nash equilibria, and collective attention dilemmas in short-form video platforms (TikTok, Instagram Reels, YouTube Shorts).

Developed for academic evaluation combining Game Theory concepts, Python computations, numerical results, and an interactive Streamlit dashboard.

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

In algorithmic social platforms, competing creators vie for finite viewer attention and recommendation distribution. Each creator faces a strategic trade-off:
1. **Trending Content ($T$):** Mimic viral formats and audio. Low barrier to entry and high short-term virality, but creates feed congestion and viewer fatigue when duplicated.
2. **Original Content ($O$):** Invest in unique, high-effort production. Builds loyal audience engagement and retention, but risks being overshadowed by viral spikes.
3. **Niche Content ($N$):** Focus on dedicated subcultures. Insulates the creator from mainstream feed cannibalization.

Game Theory allows us to formally model these strategic interactions, identify stable outcomes (Nash Equilibria), and quantify the systemic inefficiencies of attention competition.

---

## 📚 Syllabus Concept Mapping

| Syllabus Concept | Project Implementation |
| :--- | :--- |
| **Normal-Form Game** | Represent two competing content creators and their strategies in a payoff matrix. |
| **Strategies & Payoffs** | Compare Trending vs. Original (and Niche) engagement outcomes. |
| **Dominant Strategy** | Formally test whether chasing viral trends is strictly or weakly dominant. |
| **Nash Equilibrium** | Solve for pure and mixed strategy equilibria where neither creator can unilaterally improve. |
| **Pareto Optimality** | Contrast individual selfish rationality $(3, 3)$ with the social optimum $(5, 5)$. |
| **Price of Anarchy (PoA)** | Quantify the welfare destruction ($\text{PoA} \approx 1.67$, $\sim 40\%$ loss) of uncoordinated competition. |
| **Folk Theorem** | Theoretical extension explaining how repeated interaction and future rewards can sustain mutual quality. |

---

## 🧠 The Strategic Payoff Matrix (2×2)

| Creator A \ Creator B | Trending ($T$) | Original ($O$) |
|:---|:---:|:---:|
| **Trending ($T$)** | **(3.0, 3.0)** <br> *(Nash Equilibrium)* | **(6.0, 2.0)** <br> *(Viral Steal)* |
| **Original ($O$)** | **(2.0, 6.0)** <br> *(Viral Steal)* | **(5.0, 5.0)** <br> *(Social Optimum)* |

*(Payoff values represent standardized engagement units: views, watch time, shares, and retention).*

### Core Results Summary:
- **Strictly Dominant Strategy:** Trending ($T$) for both creators.
- **Unique Nash Equilibrium:** $(Trending, Trending) = (3.0, 3.0)$.
- **Social Optimum:** $(Original, Original) = (5.0, 5.0)$ with combined welfare $= 10.0$.
- **Attention Economy Prisoner's Dilemma:** Individual optimization yields $(3.0, 3.0)$, strictly dominated by $(5.0, 5.0)$.
- **Price of Anarchy (PoA):** $\text{PoA} = \frac{10.0}{6.0} \approx 1.67$ ($40\%$ welfare loss).

---

## 📖 In-Depth Presentation & Theory Guide

For the complete academic write-up, detailed mathematical proofs, derivations, scenario comparisons, and viva Q&A cheat-sheet, see:

👉 **[THEORY_GUIDE.md](file:///c:/Users/tejas/Documents/tejas/codin/Game%20Theory/THEORY_GUIDE.md)**

Included in the theory guide:
1. Executive Summary & Problem Statement
2. Formal Game-Theoretic Mathematical Definition
3. Mathematical Proofs (Best-Response, Dominance, Pure Nash, Mixed-Strategy Indifference, Pareto Dominance, Price of Anarchy)
4. Game Typology (Prisoner's Dilemma, Stag Hunt, Hawk-Dove, 3×3 Extended Model)
5. Repeated Game Extension: The Folk Theorem (Finite unravelling vs. Infinite $\delta$, Mathematical proof of $\delta^* \approx 0.33$, Axelrod's Tit-for-Tat principles)
6. Platform Mechanism Design & Policy Implications
7. Viva & Examiner Q&A Cheat-Sheet (Top 5 examiner questions & answers)

---

## 💻 Application Features

The interactive dashboard provides a unified, real-time computational environment:
1. **Interactive Payoff Matrix:** Visual indicator badges for Best Responses (`BR A` / `BR B`), Pure-Strategy Nash Equilibria (`Nash Eq`), and Social Welfare Optima (`Social Opt`).
2. **Live In-Place Matrix Editor:** Modify any payoff cell in real-time to immediately observe changes in Nash equilibria, dominance, and efficiency.
3. **Scenario Presets:** Instant loading for *Prisoner's Dilemma*, *Stag Hunt*, *Hawk-Dove*, and *3x3 Extended*.
4. **Dominance & Mixed Strategies:** Analytical tests for strictly/weakly dominant strategies and mixed-strategy Nash equilibrium derivations.
5. **Pareto Optimality & Price of Anarchy:** Quantifies efficiency loss and Price of Anarchy (PoA) resulting from selfish individual incentives.
6. **Folk Theorem Extension:** Short theoretical callout connecting one-shot competition to ongoing repeated interaction.

---

## 📂 Project Architecture

```
Game Theory/
├── app.py              # Main interactive Streamlit application with modern UI
├── game_theory.py      # Core game theory mathematical engine (Nash, BR, PoA)
├── simulation.py       # Algorithmic creator behavioral policies & simulation engine
├── THEORY_GUIDE.md     # In-depth theoretical proofs, derivations & presentation guide
├── README.md           # Project overview, quick start & summary
└── run.bat             # One-click Windows launcher
```
