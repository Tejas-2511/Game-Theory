"""
SocialMedia.Nash — Game Theory-Based Analysis of Social Media Content Competition
Interactive Streamlit Application for 10-Mark Academic Evaluation.
"""

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from game_theory import GameModel
from simulation import MultiRoundSimulator, get_agent_by_type

# Page setup
st.set_page_config(
    page_title="SocialMedia.Nash — Game Theory Simulation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Minimalist with Subtle Color Accents)
st.html("""
<style>
    /* Global Clean Font & Spacing */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Minimalist Metric Card */
    .metric-box {
        background: #121214;
        border: 1px solid #27272a;
        border-radius: 2px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .metric-title {
        color: #71717a;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .metric-sub {
        color: #71717a;
        font-size: 0.78rem;
        margin-top: 2px;
    }

    /* Minimalist Matrix Table with Subtle Color Accents */
    .matrix-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 12px;
        margin-bottom: 18px;
        border: 1px solid #27272a;
    }
    .matrix-header {
        background-color: #18181b;
        color: #e4e4e7;
        padding: 12px 10px;
        text-align: center;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid #27272a;
        border-radius: 0px;
        letter-spacing: 0.02em;
    }
    .matrix-cell {
        background: #0f0f11;
        border: 1px solid #27272a;
        border-radius: 0px;
        padding: 16px 10px;
        text-align: center;
        transition: background 0.15s ease;
    }
    .matrix-cell:hover {
        background: #18181b;
    }
    .matrix-cell.nash {
        background: #131a16;
        border: 1px solid #34d399;
    }
    .matrix-cell.social-opt {
        border: 1px dashed #fbbf24;
    }
    .payoff-a {
        color: #38bdf8;
        font-size: 1.25rem;
        font-weight: 700;
    }
    .payoff-b {
        color: #c084fc;
        font-size: 1.25rem;
        font-weight: 700;
    }
    .badge {
        display: inline-block;
        padding: 2px 6px;
        border-radius: 2px;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin: 2px 2px;
    }
    .badge-nash {
        background-color: rgba(52, 211, 153, 0.12);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.35);
    }
    .badge-social {
        background-color: rgba(251, 191, 36, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.35);
    }
    .badge-br-a {
        background-color: rgba(56, 189, 248, 0.1);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .badge-br-b {
        background-color: rgba(192, 132, 252, 0.1);
        color: #c084fc;
        border: 1px solid rgba(192, 132, 252, 0.3);
    }

    /* Clean Subtle Callout Boxes */
    .callout-box {
        background: #141416;
        border: 1px solid #27272a;
        border-left: 3px solid #38bdf8;
        border-radius: 2px;
        padding: 12px 16px;
        margin-bottom: 12px;
        font-size: 0.88rem;
        line-height: 1.6;
        color: #d4d4d8;
    }
    .callout-box.nash {
        border-left-color: #34d399;
    }
    .callout-box.dilemma {
        border-left-color: #f87171;
    }
    .callout-box.info {
        border-left-color: #38bdf8;
    }
    .callout-box.warning {
        border-left-color: #fbbf24;
    }
</style>
""")


# ==========================================
# SIDEBAR: MODEL CONFIGURATION
# ==========================================
st.sidebar.title("Model Configuration")
st.sidebar.markdown("Game parameters and payoff settings.")

game_type = st.sidebar.radio(
    "Game Complexity",
    ["2x2 Standard (Trending vs Original)", "3x3 Extended (Trending, Original, Niche)", "Custom Matrix Editor"],
    index=0
)

# Presets for 2x2
preset_choice = "Custom"
if game_type == "2x2 Standard (Trending vs Original)":
    preset_choice = st.sidebar.selectbox(
        "Game Preset Scenario",
        [
            "Prisoner's Dilemma (Viral Rat Race - Default)",
            "Stag Hunt (Mutual High Quality Rewards)",
            "Hawk-Dove / Chicken (Audience Congestion)",
            "Custom Values"
        ]
    )

# Payoffs state management
if "custom_p_a_2x2" not in st.session_state:
    st.session_state.custom_p_a_2x2 = np.array([[3.0, 6.0], [2.0, 5.0]])
    st.session_state.custom_p_b_2x2 = np.array([[3.0, 2.0], [6.0, 5.0]])

if "custom_p_a_3x3" not in st.session_state:
    st.session_state.custom_p_a_3x3 = np.array([
        [3.0, 6.0, 5.0],
        [2.0, 5.0, 4.0],
        [4.0, 4.0, 3.0]
    ])
    st.session_state.custom_p_b_3x3 = np.array([
        [3.0, 2.0, 4.0],
        [6.0, 5.0, 4.0],
        [5.0, 4.0, 3.0]
    ])

# Apply preset values
if game_type == "2x2 Standard (Trending vs Original)":
    if preset_choice == "Prisoner's Dilemma (Viral Rat Race - Default)":
        p_a = np.array([[3.0, 6.0], [2.0, 5.0]])
        p_b = np.array([[3.0, 2.0], [6.0, 5.0]])
    elif preset_choice == "Stag Hunt (Mutual High Quality Rewards)":
        p_a = np.array([[2.0, 2.0], [0.0, 5.0]])
        p_b = np.array([[2.0, 0.0], [2.0, 5.0]])
    elif preset_choice == "Hawk-Dove / Chicken (Audience Congestion)":
        p_a = np.array([[1.0, 6.0], [2.0, 4.0]])
        p_b = np.array([[1.0, 2.0], [6.0, 4.0]])
    else:  # Custom Values
        st.sidebar.markdown("### Edit Payoff Matrix (2x2)")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.caption("**Creator A Payoffs**")
            t_t_a = st.number_input("A: Trend/Trend", value=3.0, step=0.5)
            t_o_a = st.number_input("A: Trend/Orig", value=6.0, step=0.5)
            o_t_a = st.number_input("A: Orig/Trend", value=2.0, step=0.5)
            o_o_a = st.number_input("A: Orig/Orig", value=5.0, step=0.5)
            p_a = np.array([[t_t_a, t_o_a], [o_t_a, o_o_a]])
        with col2:
            st.caption("**Creator B Payoffs**")
            t_t_b = st.number_input("B: Trend/Trend", value=3.0, step=0.5)
            t_o_b = st.number_input("B: Trend/Orig", value=2.0, step=0.5)
            o_t_b = st.number_input("B: Orig/Trend", value=6.0, step=0.5)
            o_o_b = st.number_input("B: Orig/Orig", value=5.0, step=0.5)
            p_b = np.array([[t_t_b, t_o_b], [o_t_b, o_o_b]])

    game = GameModel(["Trending", "Original"], p_a, p_b)

elif game_type == "3x3 Extended (Trending, Original, Niche)":
    game = GameModel.create_3x3_extended()
else:
    # Custom Matrix Editor
    st.sidebar.markdown("### Custom 3x3 Payoff Matrix")
    strategies_custom = ["Trending", "Original", "Niche"]
    p_a_cust = np.copy(st.session_state.custom_p_a_3x3)
    p_b_cust = np.copy(st.session_state.custom_p_b_3x3)

    st.sidebar.markdown("Modify Creator A & B payoffs:")
    for i, s1 in enumerate(strategies_custom):
        for j, s2 in enumerate(strategies_custom):
            col1, col2 = st.sidebar.columns(2)
            p_a_cust[i, j] = col1.number_input(f"A: ({s1}, {s2})", value=float(p_a_cust[i, j]), step=0.5, key=f"ca_{i}_{j}")
            p_b_cust[i, j] = col2.number_input(f"B: ({s1}, {s2})", value=float(p_b_cust[i, j]), step=0.5, key=f"cb_{i}_{j}")
    game = GameModel(strategies_custom, p_a_cust, p_b_cust)


# App Header (Clean & Minimalist)
st.html("""
<div style="padding: 10px 0 18px 0; border-bottom: 1px solid #27272a; margin-bottom: 18px;">
    <h1 style="color: #ffffff; font-size: 2.0rem; font-weight: 700; margin: 0 0 6px 0; letter-spacing: -0.02em;">
        Game Theory: Social Media Content Competition
    </h1>
    <p style="color: #71717a; font-size: 0.95rem; margin: 0;">
        Strategic modeling of creator incentives, Nash equilibria, and attention dynamics.
    </p>
</div>
""")


# Top-level quick KPI strip
nash_eqs = game.find_pure_nash_equilibria()
dominant_strats = game.find_dominant_strategies()
social_optima = game.find_social_optimum()
pd_check = game.is_prisoners_dilemma()

col1, col2, col3, col4 = st.columns(4)
with col1:
    nash_desc = ", ".join([f"({game.strategies[i]}, {game.strategies[j]})" for (i, j) in nash_eqs]) if nash_eqs else "None (Pure)"
    st.html(f"""
    <div class="metric-box">
        <div class="metric-title">Nash Equilibrium</div>
        <div class="metric-value" style="color: #34d399;">{nash_desc}</div>
        <div class="metric-sub">Self-enforcing outcome</div>
    </div>
    """)

with col2:
    dom_a = dominant_strats["Creator A"]["strictly_dominant"] or "None"
    st.html(f"""
    <div class="metric-box">
        <div class="metric-title">Creator A Dominant Strategy</div>
        <div class="metric-value" style="color: #38bdf8;">{dom_a}</div>
        <div class="metric-sub">Optimal regardless of opponent</div>
    </div>
    """)

with col3:
    dom_b = dominant_strats["Creator B"]["strictly_dominant"] or "None"
    st.html(f"""
    <div class="metric-box">
        <div class="metric-title">Creator B Dominant Strategy</div>
        <div class="metric-value" style="color: #c084fc;">{dom_b}</div>
        <div class="metric-sub">Optimal regardless of opponent</div>
    </div>
    """)

with col4:
    opt_str = f"({game.strategies[social_optima[0][0]]}, {game.strategies[social_optima[0][1]]})"
    opt_val = social_optima[0][2]
    st.html(f"""
    <div class="metric-box">
        <div class="metric-title">Social Optimum</div>
        <div class="metric-value" style="color: #fbbf24;">{opt_str} = {opt_val}</div>
        <div class="metric-sub">Pareto efficient state</div>
    </div>
    """)


# ==========================================
# PAYOFF MATRIX & EQUILIBRIUM ANALYSIS
# ==========================================
st.subheader("Strategic Normal-Form Representation")
st.markdown("""
Competition for viewer attention is modeled as a 2-player strategic normal-form game.
The matrix below maps the engagement payoffs $(U_A, U_B)$ for each strategy combination.
""")

best_a = game.get_best_responses_a()
best_b = game.get_best_responses_b()
pareto_cells = set(game.find_pareto_efficient_outcomes())
social_cells = {(c[0], c[1]) for c in social_optima}

# Render HTML Matrix Table
html_table = '<table class="matrix-table">'
html_table += '<thead><tr><th class="matrix-header" style="background:#18181b; color:#71717a;">Creator A \\ Creator B</th>'
for strat in game.strategies:
    html_table += f'<th class="matrix-header"><span style="color:#e4e4e7;">Creator B: {strat}</span></th>'
html_table += '</tr></thead><tbody>'

for i, strat_a in enumerate(game.strategies):
    html_table += f'<tr><td class="matrix-header"><span style="color:#e4e4e7;">Creator A: {strat_a}</span></td>'
    for j, strat_b in enumerate(game.strategies):
        is_nash = (i, j) in nash_eqs
        is_social = (i, j) in social_cells
        is_br_a = i in best_a[j]
        is_br_b = j in best_b[i]

        cell_classes = ["matrix-cell"]
        if is_nash:
            cell_classes.append("nash")
        elif is_social:
            cell_classes.append("social-opt")

        badges = []
        if is_nash:
            badges.append('<span class="badge badge-nash">Nash Eq</span>')
        if is_social:
            badges.append('<span class="badge badge-social">Social Opt</span>')
        if is_br_a:
            badges.append('<span class="badge badge-br-a">BR A</span>')
        if is_br_b:
            badges.append('<span class="badge badge-br-b">BR B</span>')

        badge_html = f"<div style='margin-top: 6px;'>{' '.join(badges)}</div>" if badges else ""
        val_a = game.payoff_a[i, j]
        val_b = game.payoff_b[i, j]

        html_table += (
            f'<td class="{" ".join(cell_classes)}">'
            f'<span class="payoff-a">{val_a:.1f}</span>, '
            f'<span class="payoff-b">{val_b:.1f}</span>'
            f'{badge_html}'
            f'</td>'
        )
    html_table += '</tr>'
html_table += '</tbody></table>'

st.html(html_table)

# Legends (Subtle colors)
st.html("""
<div style="margin-top: 8px; margin-bottom: 18px; font-size: 0.88rem; color: #a1a1aa; line-height: 1.8;">
    <span style="font-weight: 600; color: #ffffff;">Legend:</span>
    <span style="margin-left: 12px;"><b style="color: #38bdf8;">(A)</b> Creator A Payoff</span>
    <span style="margin-left: 12px;"><b style="color: #c084fc;">(B)</b> Creator B Payoff</span>
    <span style="margin-left: 12px;"><span class="badge badge-br-a">BR A</span> Best response A</span>
    <span style="margin-left: 12px;"><span class="badge badge-br-b">BR B</span> Best response B</span>
    <span style="margin-left: 12px;"><span class="badge badge-nash">Nash Eq</span> Equilibrium</span>
    <span style="margin-left: 12px;"><span class="badge badge-social">Social Opt</span> Max Combined</span>
</div>
""")

st.markdown("---")

# Side-by-side Analytical Insights
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### Equilibrium Analysis")
    if nash_eqs:
        for (ni, nj) in nash_eqs:
            st.html(f"""
            <div class="callout-box nash">
                <b style="color: #34d399;">Pure-Strategy Nash Equilibrium at ({game.strategies[ni]}, {game.strategies[nj]})</b><br/>
                • Creator A receives <b>{game.payoff_a[ni, nj]:.1f}</b><br/>
                • Creator B receives <b>{game.payoff_b[ni, nj]:.1f}</b><br/>
                • Neither creator can unilaterally deviate to another strategy and achieve higher engagement.
            </div>
            """)
    else:
        st.html("""
        <div class="callout-box warning">
            No pure-strategy Nash equilibrium exists in this payoff configuration. Creators engage in cyclic strategy switching.
        </div>
        """)

    # Mixed Strategy calculation if 2x2
    mixed_res = game.calculate_mixed_nash_2x2()
    if mixed_res:
        st.markdown("#### Mixed-Strategy Nash Equilibrium")
        if mixed_res["is_valid_internal"]:
            st.html(f"""
            <div class="callout-box info">
                <b>Mixed Strategy Probabilities:</b><br/>
                • Creator A plays <b>Trending</b> with p = {mixed_res['p_creator_a_trending']:.2f} and <b>Original</b> with 1-p = {mixed_res['p_creator_a_original']:.2f}<br/>
                • Creator B plays <b>Trending</b> with q = {mixed_res['q_creator_b_trending']:.2f} and <b>Original</b> with 1-q = {mixed_res['q_creator_b_original']:.2f}
            </div>
            """)
        else:
            st.caption("Pure dominant strategies prevent an interior mixed-strategy equilibrium.")

with col_right:
    st.markdown("### The Attention Economy Prisoner's Dilemma")
    if pd_check["is_pd"]:
        st.html(f"""
        <div class="callout-box dilemma">
            <b style="color: #f87171;">Prisoner's Dilemma Confirmed</b><br/>
            {pd_check['reason']}<br/><br/>
            • <b>Nash Equilibrium Outcome:</b> ({game.strategies[pd_check['nash'][0]]}, {game.strategies[pd_check['nash'][1]]}) with combined payoff = <b>{game.payoff_a[pd_check['nash']]+game.payoff_b[pd_check['nash']]:.1f}</b><br/>
            • <b>Social Optimum Outcome:</b> ({game.strategies[pd_check['cooperative'][0]]}, {game.strategies[pd_check['cooperative'][1]]}) with combined payoff = <b>{game.payoff_a[pd_check['cooperative']]+game.payoff_b[pd_check['cooperative']]:.1f}</b><br/>
            • <b>Price of Anarchy (PoA):</b> <code style="color:#f4f4f5; background:#27272a; padding:2px 4px;">{pd_check['price_of_anarchy']}</code><br/>
            • <b>Lost Engagement Surplus:</b> <code style="color:#f4f4f5; background:#27272a; padding:2px 4px;">{pd_check['efficiency_loss']:.1f}</code> units of engagement lost to trend congestion
        </div>
        """)
    else:
        st.html(f"""
        <div class="callout-box info">
            <b>Game Classification:</b> {pd_check['reason']}
        </div>
        """)

    st.markdown("#### Academic Insight for Evaluation")
    st.markdown("""
    In platforms like TikTok, Instagram Reels, and YouTube Shorts:
    - When a viral trend emerges, **jumping on the trend dominates** producing original content because algorithms prioritize viral sound clips.
    - When **both creators chase the same trend**, audience attention is fragmented, causing viewer fatigue and lower combined satisfaction $(3, 3)$ instead of $(5, 5)$.
    """)

st.markdown("---")
st.subheader("Syllabus Concept Mapping")
st.html("""
<table style="width: 100%; border-collapse: collapse; margin-top: 10px; margin-bottom: 20px; font-size: 0.9rem;">
  <thead>
    <tr style="border-bottom: 1px solid #27272a; text-align: left;">
      <th style="padding: 10px 14px; color: #a1a1aa; font-weight: 600; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; width: 32%;">Syllabus concept</th>
      <th style="padding: 10px 14px; color: #a1a1aa; font-weight: 600; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em;">How we'll use it</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #1c1c1f;">
      <td style="padding: 11px 14px; color: #ffffff; font-weight: 600;">Normal-form game</td>
      <td style="padding: 11px 14px; color: #d4d4d8;">Represent two content creators and their strategies in a payoff matrix.</td>
    </tr>
    <tr style="border-bottom: 1px solid #1c1c1f;">
      <td style="padding: 11px 14px; color: #ffffff; font-weight: 600;">Strategies and payoffs</td>
      <td style="padding: 11px 14px; color: #d4d4d8;">Compare Trending vs Original content.</td>
    </tr>
    <tr style="border-bottom: 1px solid #1c1c1f;">
      <td style="padding: 11px 14px; color: #ffffff; font-weight: 600;">Dominant strategy</td>
      <td style="padding: 11px 14px; color: #d4d4d8;">Check whether one strategy is always better.</td>
    </tr>
    <tr style="border-bottom: 1px solid #1c1c1f;">
      <td style="padding: 11px 14px; color: #ffffff; font-weight: 600;">Nash equilibrium</td>
      <td style="padding: 11px 14px; color: #d4d4d8;">Find the outcome where neither creator benefits by changing strategy alone.</td>
    </tr>
    <tr>
      <td style="padding: 11px 14px; color: #ffffff; font-weight: 600;">Pareto optimality</td>
      <td style="padding: 11px 14px; color: #d4d4d8;">Compare individual incentives with the best combined outcome.</td>
    </tr>
  </tbody>
</table>
""")
