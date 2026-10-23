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
<div style="padding: 6px 0 16px 0; border-bottom: 1px solid #27272a; margin-bottom: 20px;">
    <h1 style="color: #ffffff; font-size: 1.95rem; font-weight: 700; margin: 0 0 6px 0; letter-spacing: -0.02em;">
        Game Theory: Social Media Content Competition
    </h1>
    <p style="color: #71717a; font-size: 0.92rem; margin: 0;">
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
        <div class="metric-sub">Self-enforcing mutual best response</div>
    </div>
    """)

with col2:
    dom_a = dominant_strats["Creator A"]["strictly_dominant"] or "None"
    st.html(f"""
    <div class="metric-box">
        <div class="metric-title">Creator A Dominance</div>
        <div class="metric-value" style="color: #38bdf8;">{dom_a}</div>
        <div class="metric-sub">Optimal regardless of Creator B</div>
    </div>
    """)

with col3:
    dom_b = dominant_strats["Creator B"]["strictly_dominant"] or "None"
    st.html(f"""
    <div class="metric-box">
        <div class="metric-title">Creator B Dominance</div>
        <div class="metric-value" style="color: #c084fc;">{dom_b}</div>
        <div class="metric-sub">Optimal regardless of Creator A</div>
    </div>
    """)

with col4:
    opt_str = f"({game.strategies[social_optima[0][0]]}, {game.strategies[social_optima[0][1]]})"
    opt_val = social_optima[0][2]
    st.html(f"""
    <div class="metric-box">
        <div class="metric-title">Social / Pareto Optimum</div>
        <div class="metric-value" style="color: #fbbf24;">{opt_str} = {opt_val}</div>
        <div class="metric-sub">Total audience welfare maximization</div>
    </div>
    """)


# ==========================================
# PAYOFF MATRIX & EQUILIBRIUM ANALYSIS
# ==========================================
st.subheader("Normal-Form Strategy Profiles & Payoff Matrix")
st.markdown("""
Competition for viewer attention is modeled as a 2-player strategic normal-form game:
- **Game Formulation:** `G = ⟨N, S, u⟩` where `N = {Creator A, Creator B}`
- **Strategy Sets:** `S = {Trending, Original}` (viral aggregator vs. original quality)
- **Payoffs:** Matrix cells `(u_A, u_B)` quantify audience reach, retention, and viewer engagement
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
    st.markdown("### Dominance & Nash Equilibrium Analysis")
    if nash_eqs:
        for (ni, nj) in nash_eqs:
            st.html(f"""
            <div class="callout-box nash">
                <b style="color: #34d399;">Pure-Strategy Nash Equilibrium: ({game.strategies[ni]}, {game.strategies[nj]})</b><br/>
                • <b>Condition:</b> <code>u<sub>A</sub>(s<sub>A</sub>*, s<sub>B</sub>*) ≥ u<sub>A</sub>(s<sub>A</sub>, s<sub>B</sub>*)</code> and <code>u<sub>B</sub>(s<sub>A</sub>*, s<sub>B</sub>*) ≥ u<sub>B</sub>(s<sub>A</sub>*, s<sub>B</sub>)</code><br/>
                • <b>Payoffs:</b> Creator A = <b>{game.payoff_a[ni, nj]:.1f}</b>, Creator B = <b>{game.payoff_b[ni, nj]:.1f}</b><br/>
                • <b>Stability:</b> Neither creator can unilaterally deviate to another strategy to improve reach. Given Creator B plays {game.strategies[nj]}, Creator A's best response is {game.strategies[ni]}.
            </div>
            """)
    else:
        st.html("""
        <div class="callout-box warning">
            <b>No Pure-Strategy Nash Equilibrium:</b> In this payoff configuration, best responses form a cyclical loop, resulting in perpetual strategy switching.
        </div>
        """)

    # Dominant Strategy Check
    dom_a_type = "Strictly Dominant" if dominant_strats["Creator A"]["strictly_dominant"] else ("Weakly Dominant" if dominant_strats["Creator A"]["weakly_dominant"] else "None")
    dom_b_type = "Strictly Dominant" if dominant_strats["Creator B"]["strictly_dominant"] else ("Weakly Dominant" if dominant_strats["Creator B"]["weakly_dominant"] else "None")
    st.html(f"""
    <div class="callout-box info">
        <b>Dominant Strategy Test:</b><br/>
        • <b>Condition:</b> Strategy <code>s<sub>i</sub>*</code> strictly dominates <code>s<sub>i</sub>'</code> if <code>u<sub>i</sub>(s<sub>i</sub>*, s<sub>-i</sub>) &gt; u<sub>i</sub>(s<sub>i</sub>', s<sub>-i</sub>)</code> for all opponent actions <code>s<sub>-i</sub></code>.<br/>
        • <b>Creator A:</b> <code style="color:#38bdf8;">{dom_a}</code> ({dom_a_type})<br/>
        • <b>Creator B:</b> <code style="color:#c084fc;">{dom_b}</code> ({dom_b_type})
    </div>
    """)

    # Mixed Strategy calculation if 2x2
    mixed_res = game.calculate_mixed_nash_2x2()
    if mixed_res:
        st.markdown("#### Mixed-Strategy Nash Equilibrium")
        if mixed_res["is_valid_internal"]:
            st.html(f"""
            <div class="callout-box info">
                <b>Mixed-Strategy Randomization:</b><br/>
                • Creator A plays <b>Trending</b> with probability <code>p = {mixed_res['p_creator_a_trending']:.2f}</code> and <b>Original</b> with <code>1 - p = {mixed_res['p_creator_a_original']:.2f}</code><br/>
                • Creator B plays <b>Trending</b> with probability <code>q = {mixed_res['q_creator_b_trending']:.2f}</code> and <b>Original</b> with <code>1 - q = {mixed_res['q_creator_b_original']:.2f}</code><br/>
                • These probabilities make each opponent indifferent between their pure strategies.
            </div>
            """)
        else:
            st.caption("Dominant pure strategies preclude an interior mixed-strategy equilibrium.")

with col_right:
    st.markdown("### Pareto Optimality & Welfare Inefficiency")
    if pd_check["is_pd"]:
        st.html(f"""
        <div class="callout-box dilemma">
            <b style="color: #f87171;">Attention Economy Prisoner's Dilemma Confirmed</b><br/>
            • <b>Pareto Optimality:</b> An outcome is Pareto optimal if no player can be made better off without reducing the payoff of another.<br/>
            • <b>Social / Pareto Optimum:</b> ({game.strategies[pd_check['cooperative'][0]]}, {game.strategies[pd_check['cooperative'][1]]}) with combined payoff = <b>{game.payoff_a[pd_check['cooperative']]+game.payoff_b[pd_check['cooperative']]:.1f}</b>.<br/>
            • <b>Nash Equilibrium Outcome:</b> ({game.strategies[pd_check['nash'][0]]}, {game.strategies[pd_check['nash'][1]]}) with combined payoff = <b>{game.payoff_a[pd_check['nash']]+game.payoff_b[pd_check['nash']]:.1f}</b>.<br/>
            • <b>Inefficiency:</b> The Nash Equilibrium is strictly Pareto dominated: <code>(5.0, 5.0) &gt; (3.0, 3.0)</code>. Individual rationality creates a collectively suboptimal outcome.<br/>
            • <b>Price of Anarchy (PoA):</b> <code style="color:#f4f4f5; background:#27272a; padding:2px 4px;">PoA = Social Max / Worst Nash = {pd_check['price_of_anarchy']}</code> (Efficiency loss: {pd_check['efficiency_loss']:.1f} units).
        </div>
        """)
    else:
        st.html(f"""
        <div class="callout-box info">
            <b>Pareto State:</b> {pd_check['reason']}<br/>
            Social optimum attained at ({game.strategies[social_optima[0][0]]}, {game.strategies[social_optima[0][1]]}).
        </div>
        """)

    st.markdown("#### Real-World Content Platform Application")
    st.markdown("""
    On algorithmic platforms like TikTok, Instagram Reels, and YouTube Shorts:
    - **Algorithmic Bias:** Platforms incentivize trend-jacking by rewarding viral sound clips with quick reach, making Trending dominant for individuals.
    - **Audience Congestion:** When both creators jump on identical trends, audiences experience fatigue, resulting in fragmented engagement (3.0, 3.0) instead of (5.0, 5.0).
    """)

st.markdown("---")
# ==========================================
# REPEATED-FORM GAME DYNAMICS
# ==========================================
st.subheader("Repeated-Form Game: Multi-Round Dynamic Competition")
st.markdown("""
While single-shot games trap rational creators in mutual defection (*Trending*), a **repeated-form game** extends competition over multiple sequential rounds. 
Under the **Folk Theorem**, the shadow of the future enables creators to sustain mutual cooperation (*Original*, *Original*) as a subgame perfect equilibrium through reciprocal strategies like **Tit-for-Tat**.
""")

col_sim_a, col_sim_b, col_sim_r = st.columns([1.5, 1.5, 1.2])

agent_choices = [
    "Tit-for-Tat (Reciprocal Cooperator)",
    "Always Trending (Viral Chaser)",
    "Always Original (Quality Purist)",
    "Best-Response Learner (Greedy Adaptor)"
]

with col_sim_a:
    rep_agent_a = st.selectbox("Creator A Strategy Profile:", agent_choices, index=0, key="rep_a")
with col_sim_b:
    rep_agent_b = st.selectbox("Creator B Strategy Profile:", agent_choices, index=0, key="rep_b")
with col_sim_r:
    rep_rounds = st.slider("Content Rounds:", min_value=10, max_value=50, value=25, step=5, key="rep_rounds")

if st.button("Simulate Repeated Game", type="primary", use_container_width=True):
    sim_a = get_agent_by_type(rep_agent_a, "Creator A", game.strategies)
    sim_b = get_agent_by_type(rep_agent_b, "Creator B", game.strategies)
    simulator = MultiRoundSimulator(game, sim_a, sim_b)
    rep_results = simulator.run_simulation(num_rounds=rep_rounds, noise_std=0.0, trend_fatigue_decay=0.0, seed=42)
    st.session_state.rep_results = rep_results

if "rep_results" in st.session_state:
    res = st.session_state.rep_results
    df_rounds = pd.DataFrame(res["rounds"])

    # Compact KPI cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.html(f"""
        <div class="metric-box">
            <div class="metric-title">Creator A Total Payoff</div>
            <div class="metric-value" style="color: #38bdf8;">{res['total_payoff_a']}</div>
            <div class="metric-sub">Avg: {res['avg_payoff_a']}/round</div>
        </div>
        """)
    with k2:
        st.html(f"""
        <div class="metric-box">
            <div class="metric-title">Creator B Total Payoff</div>
            <div class="metric-value" style="color: #c084fc;">{res['total_payoff_b']}</div>
            <div class="metric-sub">Avg: {res['avg_payoff_b']}/round</div>
        </div>
        """)
    with k3:
        st.html(f"""
        <div class="metric-box">
            <div class="metric-title">Cooperation Rate</div>
            <div class="metric-value" style="color: #fbbf24;">{res['social_alignment_pct']}%</div>
            <div class="metric-sub">Mutual Original content</div>
        </div>
        """)
    with k4:
        st.html(f"""
        <div class="metric-box">
            <div class="metric-title">Defection Rate</div>
            <div class="metric-value" style="color: #34d399;">{res['nash_alignment_pct']}%</div>
            <div class="metric-sub">Mutual Trend chasing</div>
        </div>
        """)

    # Compact Line Chart
    chart_data = pd.melt(
        df_rounds,
        id_vars=["round"],
        value_vars=["cum_payoff_a", "cum_payoff_b"],
        var_name="Creator",
        value_name="Cumulative Engagement"
    )
    chart_data["Creator"] = chart_data["Creator"].map({"cum_payoff_a": "Creator A", "cum_payoff_b": "Creator B"})

    rep_chart = alt.Chart(chart_data).mark_line(strokeWidth=2.5, point=True).encode(
        x=alt.X("round:Q", title="Round Number"),
        y=alt.Y("Cumulative Engagement:Q", title="Cumulative Engagement"),
        color=alt.Color("Creator:N", scale=alt.Scale(domain=["Creator A", "Creator B"], range=["#38bdf8", "#c084fc"])),
        tooltip=["round", "Creator", "Cumulative Engagement"]
    ).properties(height=240)

    st.altair_chart(rep_chart, use_container_width=True)

    # Short Folk Theorem Insight Box
    if res['social_alignment_pct'] > 50:
        st.html("""
        <div class="callout-box nash">
            <b style="color: #34d399;">Repeated Game Dynamic (Folk Theorem):</b> By playing reciprocal strategies like Tit-for-Tat, creators successfully overcome the single-shot Prisoner's Dilemma. The discounted future reward of sustained cooperation (5.0, 5.0) outweighs the short-term temptation to defect to Trending (6.0).
        </div>
        """)
    else:
        st.html("""
        <div class="callout-box dilemma">
            <b style="color: #f87171;">Defection Trap (Rat Race):</b> Without reciprocal deterrence, the dominant strategy of chasing viral trends pulls creators into the inefficient Nash equilibrium (3.0, 3.0), resulting in mutual attention fatigue.
        </div>
        """)



