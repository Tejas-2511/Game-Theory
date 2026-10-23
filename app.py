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
# TABS INTERFACE
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Payoff Matrix & Equilibrium",
    "Single-Round Sandbox",
    "Multi-Round Simulation",
    "Human vs AI Arena",
    "Theory & Viva Guide"
])


# ==========================================
# TAB 1: PAYOFF MATRIX & EQUILIBRIUM
# ==========================================
with tab1:
    st.subheader("1. Strategic Normal-Form Representation")
    st.markdown("""
    In game theory, competition for user views, watch time, and retention is represented as a normal-form game.
    The table below maps the engagement payoff $(U_A, U_B)$ for each strategy combination.
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
        <span style="margin-left: 12px;"><b style="color: #38bdf8;">(A)</b> Creator A</span>
        <span style="margin-left: 12px;"><b style="color: #c084fc;">(B)</b> Creator B</span>
        <span style="margin-left: 12px;"><span class="badge badge-br-a">BR A</span> Best response A</span>
        <span style="margin-left: 12px;"><span class="badge badge-br-b">BR B</span> Best response B</span>
        <span style="margin-left: 12px;"><span class="badge badge-nash">Nash Eq</span> Equilibrium</span>
        <span style="margin-left: 12px;"><span class="badge badge-social">Social Opt</span> Max Combined</span>
    </div>
    """)

    st.markdown("---")

    # In-Depth Theoretical Analysis Cards
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
                No pure-strategy Nash equilibrium exists in this payoff configuration. Creators would engage in cyclic strategy switching.
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
        In real platforms like TikTok, Instagram Reels, and YouTube Shorts:
        - When a viral dance or audio trend emerges, **jumping on the trend dominates** producing high-effort original content because feeds prioritize viral sound clips.
        - But when **all creators chase the same trend**, audience attention is fragmented, leading to viewer fatigue and lower combined satisfaction $(3, 3)$ instead of $(5, 5)$.
        """)


# ==========================================
# TAB 2: SINGLE-ROUND SANDBOX
# ==========================================
with tab2:
    st.subheader("2. Real-Time Strategy Selection & Outcome Calculator")
    st.markdown("Choose individual actions for both creators and inspect immediate payoff consequences.")

    col_pick_a, col_pick_b = st.columns(2)
    with col_pick_a:
        choice_a = st.selectbox("Creator A Strategy:", game.strategies, index=0)
    with col_pick_b:
        choice_b = st.selectbox("Creator B Strategy:", game.strategies, index=1 if len(game.strategies) > 1 else 0)

    idx_a = game.strategies.index(choice_a)
    idx_b = game.strategies.index(choice_b)

    pay_a = game.payoff_a[idx_a, idx_b]
    pay_b = game.payoff_b[idx_a, idx_b]
    tot_pay = pay_a + pay_b

    is_nash_cell = (idx_a, idx_b) in nash_eqs
    is_social_cell = (idx_a, idx_b) in social_cells

    # Display Result Cards
    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        st.html(f"""
        <div class="metric-box">
            <div class="metric-title">Creator A Engagement</div>
            <div class="metric-value" style="color: #38bdf8;">{pay_a:.1f}</div>
            <div class="metric-sub">Strategy: {choice_a}</div>
        </div>
        """)
    with res_col2:
        st.html(f"""
        <div class="metric-box">
            <div class="metric-title">Creator B Engagement</div>
            <div class="metric-value" style="color: #c084fc;">{pay_b:.1f}</div>
            <div class="metric-sub">Strategy: {choice_b}</div>
        </div>
        """)
    with res_col3:
        st.html(f"""
        <div class="metric-box">
            <div class="metric-title">Combined Platform Engagement</div>
            <div class="metric-value" style="color: #fbbf24;">{tot_pay:.1f}</div>
            <div class="metric-sub">{'Social Optimum' if is_social_cell else 'Suboptimal Allocation'}</div>
        </div>
        """)

    # Comparative Bar Chart (Subtle colors, sharp bars)
    df_chart = pd.DataFrame({
        "Participant": ["Creator A", "Creator B", "Combined Total"],
        "Engagement Score": [pay_a, pay_b, tot_pay],
        "Color": ["#38bdf8", "#c084fc", "#fbbf24"]
    })

    chart = alt.Chart(df_chart).mark_bar(cornerRadius=0, size=46).encode(
        x=alt.X("Participant:N", title=None, sort=None),
        y=alt.Y("Engagement Score:Q", title="Engagement Units"),
        color=alt.Color("Color:N", scale=None),
        tooltip=["Participant", "Engagement Score"]
    ).properties(height=260)

    st.altair_chart(chart, use_container_width=True)

    # Strategic Diagnosis
    st.markdown("### Strategic Diagnosis")
    if is_nash_cell:
        st.html(f"""
        <div class="callout-box nash">
            <b style="color: #34d399;">Equilibrium State:</b> ({choice_a}, {choice_b}) is a Nash equilibrium. Neither creator has any incentive to unilaterally defect.
        </div>
        """)
    else:
        deviations = []
        # Check if A can improve
        curr_a = pay_a
        best_rows = best_a[idx_b]
        if idx_a not in best_rows:
            better_strat = game.strategies[best_rows[0]]
            better_gain = game.payoff_a[best_rows[0], idx_b] - curr_a
            deviations.append(f"• <b>Creator A</b> can gain <b>+{better_gain:.1f}</b> by deviating to <b>{better_strat}</b>.")

        # Check if B can improve
        curr_b = pay_b
        best_cols = best_b[idx_a]
        if idx_b not in best_cols:
            better_strat_b = game.strategies[best_cols[0]]
            better_gain_b = game.payoff_b[idx_a, best_cols[0]] - curr_b
            deviations.append(f"• <b>Creator B</b> can gain <b>+{better_gain_b:.1f}</b> by deviating to <b>{better_strat_b}</b>.")

        dev_html = "<br/>".join(deviations)
        st.html(f"""
        <div class="callout-box warning">
            <b style="color: #fbbf24;">Unstable Outcome:</b> This state is not an equilibrium because:<br/>
            {dev_html}
        </div>
        """)


# ==========================================
# TAB 3: ITERATED MULTI-ROUND SIMULATION
# ==========================================
with tab3:
    st.subheader("3. Dynamic Iterated Tournament Simulation")
    st.markdown("""
    In repeated interactions, content creators learn, retaliate, and adapt.
    Simulate up to **100 consecutive rounds** of social media content releases between automated behavioral agents.
    """)

    sim_col1, sim_col2, sim_col3 = st.columns(3)
    agent_options = [
        "Always Trending (Viral Chaser)",
        "Always Original (Quality Purist)",
        "Tit-for-Tat (Reciprocal Cooperator)",
        "Best-Response Learner (Greedy Adaptor)",
        "Fictitious Play (Belief-Based Strategy)",
        "Q-Learning AI (Reinforcement Agent)",
        "Random / Mixed Strategy"
    ]
    if len(game.strategies) > 2:
        agent_options.insert(2, "Always Niche (Community Specialist)")

    with sim_col1:
        agent_a_type = st.selectbox("Creator A Strategy Profile:", agent_options, index=2)
    with sim_col2:
        agent_b_type = st.selectbox("Creator B Strategy Profile:", agent_options, index=3)
    with sim_col3:
        num_rounds = st.slider("Number of Content Cycles (Rounds):", min_value=10, max_value=100, value=35, step=5)

    with st.expander("Advanced Platform Dynamics (Algorithm Noise & Fatigue)"):
        env_col1, env_col2, env_col3 = st.columns(3)
        with env_col1:
            noise_std = st.slider("Platform Algorithm Volatility (Noise σ):", min_value=0.0, max_value=1.5, value=0.2, step=0.1)
        with env_col2:
            trend_decay = st.slider("Audience Trend Fatigue Penalty (% per streak):", min_value=0.0, max_value=0.2, value=0.05, step=0.01)
        with env_col3:
            sim_seed = st.number_input("Random Seed (for reproducibility):", value=42, step=1)

    if st.button("Run Simulation", type="primary", use_container_width=True):
        agent_a = get_agent_by_type(agent_a_type, "Creator A", game.strategies)
        agent_b = get_agent_by_type(agent_b_type, "Creator B", game.strategies)

        simulator = MultiRoundSimulator(game, agent_a, agent_b)
        sim_results = simulator.run_simulation(
            num_rounds=num_rounds,
            noise_std=noise_std,
            trend_fatigue_decay=trend_decay,
            seed=int(sim_seed)
        )
        st.session_state.sim_results = sim_results

    if "sim_results" in st.session_state:
        res = st.session_state.sim_results
        df_rounds = pd.DataFrame(res["rounds"])

        # Summary KPIs
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Tournament Winner", res["winner"])
        kpi2.metric("Creator A Total Score", f"{res['total_payoff_a']}")
        kpi3.metric("Creator B Total Score", f"{res['total_payoff_b']}")
        kpi4.metric("Nash Equilibrium Rate", f"{res['nash_alignment_pct']}%")
        kpi5.metric("Social Optimum Rate", f"{res['social_alignment_pct']}%")

        # Interactive Cumulative Trajectory Chart
        st.markdown("#### Cumulative Engagement Trajectory")
        chart_data = pd.melt(
            df_rounds,
            id_vars=["round"],
            value_vars=["cum_payoff_a", "cum_payoff_b"],
            var_name="Creator",
            value_name="Cumulative Engagement"
        )
        chart_data["Creator"] = chart_data["Creator"].map({"cum_payoff_a": "Creator A", "cum_payoff_b": "Creator B"})

        cum_chart = alt.Chart(chart_data).mark_line(strokeWidth=2.5, point=True).encode(
            x=alt.X("round:Q", title="Round Number"),
            y=alt.Y("Cumulative Engagement:Q", title="Cumulative Engagement Score"),
            color=alt.Color("Creator:N", scale=alt.Scale(domain=["Creator A", "Creator B"], range=["#38bdf8", "#c084fc"])),
            tooltip=["round", "Creator", "Cumulative Engagement"]
        ).properties(height=320)

        st.altair_chart(cum_chart, use_container_width=True)

        # Strategy Selection Breakdown
        st.markdown("#### Strategy Adoption Distribution")
        strat_col1, strat_col2 = st.columns(2)

        with strat_col1:
            strat_counts_a = df_rounds["strategy_a"].value_counts().reset_index()
            strat_counts_a.columns = ["Strategy", "Count"]
            bar_a = alt.Chart(strat_counts_a).mark_bar(cornerRadius=0).encode(
                x=alt.X("Strategy:N"),
                y=alt.Y("Count:Q", title="Frequency"),
                color=alt.value("#38bdf8"),
                tooltip=["Strategy", "Count"]
            ).properties(title="Creator A Strategy Frequency", height=230)
            st.altair_chart(bar_a, use_container_width=True)

        with strat_col2:
            strat_counts_b = df_rounds["strategy_b"].value_counts().reset_index()
            strat_counts_b.columns = ["Strategy", "Count"]
            bar_b = alt.Chart(strat_counts_b).mark_bar(cornerRadius=0).encode(
                x=alt.X("Strategy:N"),
                y=alt.Y("Count:Q", title="Frequency"),
                color=alt.value("#c084fc"),
                tooltip=["Strategy", "Count"]
            ).properties(title="Creator B Strategy Frequency", height=230)
            st.altair_chart(bar_b, use_container_width=True)

        # Round History Table
        with st.expander("View Round Log & Export Data"):
            st.dataframe(df_rounds[["round", "strategy_a", "strategy_b", "payoff_a", "payoff_b", "cum_payoff_a", "cum_payoff_b", "is_nash", "is_social_optimum"]], use_container_width=True)
            csv = df_rounds.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Simulation CSV",
                data=csv,
                file_name="social_media_game_theory_simulation.csv",
                mime="text/csv"
            )


# ==========================================
# TAB 4: HUMAN VS AI CREATOR ARENA
# ==========================================
with tab4:
    st.subheader("4. Play Against AI Competitor")
    st.markdown("""
    Step into the shoes of **Creator A**! Choose your content strategy round-by-round against an adaptive AI algorithm (**Creator B**)
    and see if you fall into the viral rat race or succeed in establishing mutual cooperation.
    """)

    if "arena_history" not in st.session_state:
        st.session_state.arena_history = []
        st.session_state.arena_score_user = 0.0
        st.session_state.arena_score_ai = 0.0

    arena_ai_model = st.selectbox(
        "Choose AI Competitor Style:",
        ["Tit-for-Tat (Reciprocal)", "Fictitious Play (Belief Learning)", "Greedy Best-Response", "Pure Viral Chaser"],
        key="arena_ai_choice"
    )

    arena_col_play, arena_col_stats = st.columns([1, 1])

    with arena_col_play:
        st.markdown("#### Choose Your Next Post Type:")
        user_choice = st.radio("Content Strategy:", game.strategies, horizontal=True)

        col_btn1, col_btn2 = st.columns(2)
        submit_post = col_btn1.button("Publish Post", type="primary")
        reset_game = col_btn2.button("Reset Match")

        if reset_game:
            st.session_state.arena_history.clear()
            st.session_state.arena_score_user = 0.0
            st.session_state.arena_score_ai = 0.0
            st.rerun()

        if submit_post:
            round_idx = len(st.session_state.arena_history)
            user_act = game.strategies.index(user_choice)

            # Determine AI choice based on selected style
            if arena_ai_model == "Pure Viral Chaser":
                ai_act = 0
            elif arena_ai_model == "Tit-for-Tat (Reciprocal)":
                if round_idx == 0:
                    ai_act = 1 if len(game.strategies) > 1 else 0  # Start with Original
                else:
                    ai_act = st.session_state.arena_history[-1]["user_act"]
            elif arena_ai_model == "Greedy Best-Response":
                if round_idx == 0:
                    ai_act = 0
                else:
                    last_user = st.session_state.arena_history[-1]["user_act"]
                    ai_act = game.get_best_responses_b().get(last_user, [0])[0]
            else:  # Fictitious Play
                if round_idx == 0:
                    ai_act = 0
                else:
                    past_user_acts = [h["user_act"] for h in st.session_state.arena_history]
                    counts = np.bincount(past_user_acts, minlength=len(game.strategies))
                    probs = counts / np.sum(counts)
                    exp_pay = [np.dot(game.payoff_b[:, a_idx], probs) for a_idx in range(len(game.strategies))]
                    ai_act = int(np.argmax(exp_pay))

            pay_user = game.payoff_a[user_act, ai_act]
            pay_ai = game.payoff_b[user_act, ai_act]

            st.session_state.arena_score_user += pay_user
            st.session_state.arena_score_ai += pay_ai

            st.session_state.arena_history.append({
                "round": round_idx + 1,
                "user_choice": game.strategies[user_act],
                "ai_choice": game.strategies[ai_act],
                "user_act": user_act,
                "ai_act": ai_act,
                "pay_user": pay_user,
                "pay_ai": pay_ai
            })
            st.rerun()

    with arena_col_stats:
        st.markdown("#### Live Arena Scoreboard")
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Your Total Score", f"{st.session_state.arena_score_user:.1f}")
        mcol2.metric("AI Total Score", f"{st.session_state.arena_score_ai:.1f}")
        mcol3.metric("Rounds Completed", len(st.session_state.arena_history))

        if st.session_state.arena_history:
            last_round = st.session_state.arena_history[-1]
            st.info(
                f"**Last Round #{last_round['round']} Results:**\n\n"
                f"- You posted: **{last_round['user_choice']}** (+{last_round['pay_user']:.1f} pts)\n"
                f"- AI posted: **{last_round['ai_choice']}** (+{last_round['pay_ai']:.1f} pts)"
            )

            # Coaching feedback
            if last_round["user_choice"] == "Trending" and last_round["ai_choice"] == "Trending":
                st.warning("**Viral Cannibalization:** Both of you jumped on the same trend, diluting feed reach to (3.0, 3.0).")
            elif last_round["user_choice"] == "Original" and last_round["ai_choice"] == "Original":
                st.success("**Creative Synergy:** Both invested in original content, maximizing combined engagement (5.0, 5.0).")
            elif last_round["user_choice"] == "Trending" and last_round["ai_choice"] == "Original":
                st.info("**Viral Advantage:** You capitalized on the trend while AI built high-effort original content, granting you peak views (+6.0).")


# ==========================================
# TAB 5: PROJECT REPORT & VIVA GUIDE
# ==========================================
with tab5:
    st.subheader("5. Academic Project Report")
    st.markdown("""
    This section synthesizes the complete academic write-up for your project presentation and viva examination.
    """)

    st.markdown("""
    ### 1. Problem Statement
    In modern algorithmic social platforms (TikTok, Instagram Reels, YouTube Shorts), content creators compete for finite viewer attention.
    Each creator faces a fundamental strategic dilemma:
    - **Follow Trending Topics ($T$):** Low production barrier, high viral probability, but results in audience fatigue and feed congestion if replicated.
    - **Produce Original Content ($O$):** High production effort, deeper loyalty and audience retention, but vulnerability to being overshadowed by viral spikes.

    ### 2. Mathematical Formalization
    We model the competition as a 2-player strategic normal-form game:
    $$\\Gamma = \\langle \\mathcal{N}, (\\mathcal{S}_i)_{i \\in \\mathcal{N}}, (u_i)_{i \\in \\mathcal{N}} \\rangle$$
    Where:
    - $\\mathcal{N} = \\{ \\text{Creator A}, \\text{Creator B} \\}$
    - Strategy sets: $\\mathcal{S}_A = \\mathcal{S}_B = \\{ \\text{Trending}, \\text{Original} \\}$
    - Payoffs $u_i(s_A, s_B)$ represent engagement scores derived from:
      $$\\text{Engagement} = \\alpha \\cdot \\text{Views} + \\beta \\cdot \\text{Likes} + \\gamma \\cdot \\text{Retention}$$

    ### 3. Payoff Matrix & Prisoner's Dilemma Demonstration
    """)

    st.table(pd.DataFrame({
        "Creator A \\ B": ["Trending (T)", "Original (O)"],
        "Trending (T)": ["(3, 3) — Nash Eq", "(6, 2) — Exploitation"],
        "Original (O)": ["(2, 6) — Exploitation", "(5, 5) — Social Optimum"]
    }).set_index("Creator A \\ B"))

    st.markdown("""
    ### 4. Equilibrium Proof & Strategic Dominance
    - **For Creator A:**
      - If B plays Trending: $u_A(T, T) = 3 > u_A(O, T) = 2$ $\\Rightarrow$ Trending is best response.
      - If B plays Original: $u_A(T, O) = 6 > u_A(O, O) = 5$ $\\Rightarrow$ Trending is best response.
      - Therefore, **Trending is a strictly dominant strategy** for Creator A.
    - **By symmetry**, **Trending is also strictly dominant** for Creator B.
    - **Unique Nash Equilibrium:** $(Trending, Trending)$ with payoff vector $(3, 3)$.

    ### 5. Inefficiency & Price of Anarchy (PoA)
    - **Social Welfare Maximization (Pareto Optimum):** $(Original, Original)$ yields total utility $W(O, O) = 5 + 5 = 10$.
    - **Equilibrium Welfare:** $W(T, T) = 3 + 3 = 6$.
    - **Price of Anarchy (PoA):**
      $$\\text{PoA} = \\frac{\\max_{s} W(s)}{W(s^*)} = \\frac{10}{6} \\approx 1.67$$
    This demonstrates that unregulated individual competition leads to a **40% destruction of total potential attention value** due to viral rat races.

    ### 6. Dynamic Games & Repeated Competition
    In iterated rounds (as demonstrated in Tab 3), using the **Folk Theorem**:
    - If creators care sufficiently about future payoffs (discount factor $\\delta > 0.5$), **cooperative strategies like Tit-for-Tat** can sustain $(Original, Original)$ through threat of mutual trend saturation!

    ### 7. Viva Q&A Cheat-Sheet (Common Professor Questions)
    """)

    with st.expander("Q1: Why is (Trending, Trending) a Nash Equilibrium if (Original, Original) gives higher payoffs?"):
        st.markdown("""
        **Answer:** A Nash equilibrium requires that *no player has a unilateral incentive to deviate*.
        At $(Original, Original) = (5, 5)$, if Creator A unilaterally switches to *Trending*, their payoff jumps to $6$.
        Because each creator seeks to maximize their own individual engagement, both are drawn into defection, landing at $(3, 3)$ where neither can improve by deviating alone.
        """)

    with st.expander("Q2: What happens when we introduce a third strategy like 'Niche Audience'?"):
        st.markdown("""
        **Answer:** In the 3x3 model, *Niche* provides audience differentiation. Even when a rival plays Trending, a Niche creator retains a loyal subculture payoff $(4)$, preventing total feed extinction and mitigating the harshness of the Prisoner's Dilemma.
        """)

    with st.expander("Q3: How do platforms influence this equilibrium?"):
        st.markdown("""
        **Answer:** Platform recommendation algorithms dictate the payoff matrix. If platforms introduce algorithmic fatigue penalties or reward original creation with higher distribution coefficients, they can alter payoffs such that $(Original, Original)$ becomes the dominant Nash equilibrium!
        """)
