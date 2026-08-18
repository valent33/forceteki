"""
visu.py — Visualization and analysis utilities for SWU RL training postmortem.
Import from notebooks or the Dash app to keep code centralized.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

px_defaults_set = False
try:
    import plotly.express as px
    px.defaults.template = "plotly_white"
    px_defaults_set = True
except Exception:
    pass


# ── helpers ──────────────────────────────────────────────────────────

def rolling_mean(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window=window, min_periods=max(2, window // 2)).mean()


def find_run_dir(run_name: str, base: Path | None = None) -> Path:
    """Search upwards from *base* for runs/<run_name>."""
    base = base or Path.cwd()
    for root in [base, *base.parents]:
        candidate = root / "runs" / run_name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"runs/{run_name} not found from {base}")


# ── data loading ─────────────────────────────────────────────────────

def load_episode_summaries(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "episode_summaries.csv"
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Empty file: {csv_path}")

    df["training_order"] = np.arange(1, len(df) + 1)
    df["episode"] = pd.to_numeric(df["episode"], errors="coerce")

    numeric_cols = [
        "steps", "agent_turns", "opponent_turns",
        "agent_rewards", "opponent_rewards", "total_rewards", "total_reward_steps",
        "valid_actions_sum", "valid_actions_count",
        "agent_valid_actions_sum", "agent_valid_actions_count",
        "agent_max_valid_actions",
        "regroup_segment_count", "regroup_action_total",
        "regroup_card_action_total", "regroup_agent_action_total", "regroup_opponent_action_total",
        "cards_played",
        "agent_base_hp_sum", "opp_base_hp_sum",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["reward_margin"] = df["agent_rewards"] - df["opponent_rewards"]

    # Determine win from actual winner column; fall back to reward margin proxy
    if "winner" in df.columns:
        winner_col = df["winner"].astype(str).str.lower().str.strip()
        df["agent_win"] = winner_col.isin({"agent", "111", "player1", "p1"})
        # fallback: if winner is None/unresolved/draw, use reward margin > 0
        unresolved = winner_col.isin({"none", "nan", "", "draw", "unresolved"})
        df.loc[unresolved, "agent_win"] = df.loc[unresolved, "reward_margin"] > 0
    else:
        df["agent_win"] = df["reward_margin"] > 0

    # avg valid actions computed on the fly
    df["avg_valid_actions"] = df["valid_actions_sum"] / np.maximum(1, df["valid_actions_count"])
    df["avg_agent_valid_actions"] = df["agent_valid_actions_sum"] / np.maximum(1, df["agent_valid_actions_count"])

    return df


def load_turn_summaries(run_dir: Path, ep_df: pd.DataFrame | None = None) -> pd.DataFrame:
    csv_path = run_dir / "turn_summaries.csv"
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Empty file: {csv_path}")

    df["episode"] = pd.to_numeric(df["episode"], errors="coerce")
    df["turn_number"] = pd.to_numeric(df["turn_number"], errors="coerce")

    # merge training order from episode summaries
    if ep_df is not None and "episode" in ep_df.columns and "training_order" in ep_df.columns:
        order_map = ep_df[["episode", "training_order"]].dropna().drop_duplicates("episode")
        order_map["episode"] = order_map["episode"].astype(int)
        df = df.merge(order_map, on="episode", how="left")
    if "training_order" not in df.columns or df["training_order"].isna().all():
        df["training_order"] = np.arange(1, len(df) + 1)
    else:
        df["training_order"] = df["training_order"].fillna(df.index + 1)

    # drop turn‑0 setup rows
    if df["turn_number"].min() == 0:
        df = df[df["turn_number"] > 0]

    stat_cols = [c for c in df.columns
                 if c not in {"run_id", "episode", "turn_number", "step_index", "training_order"}]
    for c in stat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ── metric metadata ──────────────────────────────────────────────────

TURN_METRICS: dict[str, tuple[str, str]] = {
    "agent_board_power":      ("Agent Board Power",      "#0891b2"),
    "opp_board_power":        ("Opponent Board Power",    "#f59e0b"),
    "agent_board_hp":         ("Agent Board HP",          "#06b6d4"),
    "opp_board_hp":           ("Opponent Board HP",       "#d97706"),
    "agent_unit_count":       ("Agent Units",             "#22c55e"),
    "opp_unit_count":         ("Opponent Units",          "#ef4444"),
    "agent_exhausted_count":  ("Agent Exhausted",         "#dc2626"),
    "opp_exhausted_count":    ("Opponent Exhausted",      "#f97316"),
    "agent_ready_resources":  ("Agent Ready Resources",   "#14b8a6"),
    "agent_credits":          ("Agent Credits",           "#f59e0b"),
    "agent_hand_count":       ("Agent Hand Size",         "#a855f7"),
    "opp_hand_count":         ("Opponent Hand Size",      "#ec4899"),
    "agent_base_hp":          ("Agent Base HP",           "#3b82f6"),
    "opp_base_hp":            ("Opponent Base HP",        "#dc2626"),
    "agent_leader_hp":        ("Agent Leader HP",         "#6366f1"),
    "opp_leader_hp":          ("Opponent Leader HP",      "#e11d48"),
    "agent_board_damage":     ("Agent Board Damage",      "#94a3b8"),
    "opp_board_damage":       ("Opponent Board Damage",   "#78716c"),
}


# ── episode‑level plots ──────────────────────────────────────────────

def plot_win_rate(df: pd.DataFrame, window: int = 20) -> go.Figure:
    x = df["training_order"]
    w = df["agent_win"].astype(float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=w, mode="markers", name="Win (1/0)",
                             marker=dict(size=4, color="#94a3b8"), opacity=0.35))
    fig.add_trace(go.Scatter(x=x, y=rolling_mean(w, window), mode="lines",
                             name=f"Rolling ({window})", line=dict(color="#22c55e", width=3)))
    fig.update_layout(title="Win Rate Over Training", height=400,
                      xaxis_title="Training order", yaxis_title="Win rate",
                      legend=dict(orientation="h", y=1.12))
    return fig


def plot_rewards(df: pd.DataFrame, window: int = 20) -> go.Figure:
    x = df["training_order"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=df["agent_rewards"], mode="markers",
                             name="Agent reward", marker=dict(size=3, color="#2563eb"), opacity=0.3))
    fig.add_trace(go.Scatter(x=x, y=rolling_mean(df["agent_rewards"], window), mode="lines",
                             name=f"Agent (w={window})", line=dict(color="#1d4ed8", width=3)))
    fig.add_trace(go.Scatter(x=x, y=df["reward_margin"], mode="markers",
                             name="Margin", marker=dict(size=3, color="#f97316"), opacity=0.25))
    fig.add_trace(go.Scatter(x=x, y=rolling_mean(df["reward_margin"], window), mode="lines",
                             name=f"Margin (w={window})", line=dict(color="#ea580c", width=3, dash="dot")))
    fig.update_layout(title="Reward Trends", height=400,
                      xaxis_title="Training order", yaxis_title="Reward",
                      legend=dict(orientation="h", y=1.12))
    return fig


def plot_episode_length(df: pd.DataFrame, window: int = 20) -> go.Figure:
    x = df["training_order"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=df["steps"], mode="markers", name="Steps",
                             marker=dict(size=3, color="#0f766e"), opacity=0.3))
    fig.add_trace(go.Scatter(x=x, y=rolling_mean(df["steps"], window), mode="lines",
                             name=f"Steps (w={window})", line=dict(color="#115e59", width=3)))
    fig.add_trace(go.Scatter(x=x, y=df["agent_turns"], mode="markers", name="Agent turns",
                             marker=dict(size=3, color="#14b8a6"), opacity=0.25))
    fig.add_trace(go.Scatter(x=x, y=rolling_mean(df["agent_turns"], window), mode="lines",
                             name=f"Turns (w={window})", line=dict(color="#0d9488", width=3, dash="dot")))
    fig.update_layout(title="Episode Length", height=400,
                      xaxis_title="Training order", yaxis_title="Count",
                      legend=dict(orientation="h", y=1.12))
    return fig


def plot_cards_played(df: pd.DataFrame, window: int = 20) -> go.Figure | None:
    if "cards_played" not in df.columns:
        return None
    x = df["training_order"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=df["cards_played"], mode="markers", name="Cards played",
                             marker=dict(size=3, color="#7c3aed"), opacity=0.3))
    fig.add_trace(go.Scatter(x=x, y=rolling_mean(df["cards_played"], window), mode="lines",
                             name=f"Cards (w={window})", line=dict(color="#6d28d9", width=3)))
    fig.update_layout(title="Cards Played Per Episode", height=350,
                      xaxis_title="Training order", yaxis_title="Cards",
                      legend=dict(orientation="h", y=1.12))
    return fig


def plot_action_economy(df: pd.DataFrame, window: int = 20) -> go.Figure:
    x = df["training_order"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=rolling_mean(df["avg_valid_actions"], window),
                             mode="lines", name=f"All valid actions (w={window})",
                             line=dict(color="#7c3aed", width=3)))
    fig.add_trace(go.Scatter(x=x, y=rolling_mean(df["avg_agent_valid_actions"], window),
                             mode="lines", name=f"Agent valid actions (w={window})",
                             line=dict(color="#a855f7", width=3, dash="dot")))
    fig.update_layout(title="Action Economy", height=350,
                      xaxis_title="Training order", yaxis_title="Average actions",
                      legend=dict(orientation="h", y=1.12))
    return fig


# ── turn‑level plots ─────────────────────────────────────────────────

def plot_turn_metric(turn_df: pd.DataFrame, turn: int, metric: str,
                     window: int = 10) -> go.Figure | None:
    """Single metric at a fixed turn across training."""
    sub = turn_df[turn_df["turn_number"] == turn].sort_values("training_order")
    if sub.empty or metric not in sub.columns:
        return None
    label, color = TURN_METRICS.get(metric, (metric, "#888888"))
    x = sub["training_order"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=sub[metric], mode="markers", name=label,
                             marker=dict(size=4, color=color), opacity=0.35))
    fig.add_trace(go.Scatter(x=x, y=rolling_mean(sub[metric], window), mode="lines",
                             name=f"{label} (w={window})", line=dict(color=color, width=3)))
    fig.update_layout(title=f"Turn {turn} — {label} Across Training", height=350,
                      xaxis_title="Training order", yaxis_title=label,
                      legend=dict(orientation="h", y=1.12))
    return fig


def plot_turn_dashboard(turn_df: pd.DataFrame, turn: int,
                        metrics: list[str] | None = None,
                        window: int = 10) -> go.Figure:
    """Multi‑panel dashboard for a chosen turn."""
    if metrics is None:
        metrics = ["agent_unit_count", "opp_unit_count", "agent_exhausted_count",
                   "agent_ready_resources", "agent_hand_count",
                   "agent_base_hp", "opp_base_hp",
                   "agent_board_power", "opp_board_power"]

    n = len(metrics)
    cols = 3
    rows = (n + cols - 1) // cols

    sub = turn_df[turn_df["turn_number"] == turn].sort_values("training_order")
    if sub.empty:
        return go.Figure()

    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[TURN_METRICS.get(m, (m, ""))[0] for m in metrics],
                        vertical_spacing=0.10, horizontal_spacing=0.06)
    x = sub["training_order"]
    for i, m in enumerate(metrics):
        if m not in sub.columns:
            continue
        r, c = i // cols + 1, i % cols + 1
        label, color = TURN_METRICS.get(m, (m, "#888"))
        fig.add_trace(go.Scatter(x=x, y=sub[m], mode="markers", name=label,
                                 marker=dict(size=3, color=color), opacity=0.3,
                                 showlegend=False), row=r, col=c)
        fig.add_trace(go.Scatter(x=x, y=rolling_mean(sub[m], window), mode="lines",
                                 name=f"{label} avg", line=dict(color=color, width=2),
                                 showlegend=False), row=r, col=c)

    fig.update_layout(height=280 * rows,
                      title_text=f"Turn {turn} — Dashboard",
                      template="plotly_white")
    return fig


def plot_episode_turns(turn_df: pd.DataFrame, episode: int) -> go.Figure:
    """Turn‑by‑turn breakdown for a single episode."""
    ep = turn_df[turn_df["episode"] == episode].sort_values("turn_number")
    if ep.empty:
        raise ValueError(f"No turn data for episode {episode}")
    ep = ep.groupby("turn_number", as_index=False).last()

    x = ep["turn_number"]
    fig = make_subplots(rows=2, cols=2,
        subplot_titles=("Units & Exhaustion", "Resources & Hand",
                        "Board Power", "Base & Leader HP"),
        vertical_spacing=0.15)

    # panel 1: units
    for col, label, color, dash in [
        ("agent_unit_count", "Agent units", "#22c55e", None),
        ("opp_unit_count", "Opp units", "#ef4444", None),
        ("agent_exhausted_count", "Agent exhausted", "#f97316", "dot"),
    ]:
        if col in ep.columns:
            fig.add_trace(go.Scatter(x=x, y=ep[col], mode="lines+markers",
                                     name=label, line=dict(color=color, width=2, dash=dash)),
                          row=1, col=1)

    # panel 2: resources + hand
    for col, label, color in [
        ("agent_ready_resources", "Ready resources", "#14b8a6"),
        ("agent_hand_count", "Hand size", "#a855f7"),
    ]:
        if col in ep.columns:
            fig.add_trace(go.Scatter(x=x, y=ep[col], mode="lines+markers",
                                     name=label, line=dict(color=color, width=2)),
                          row=1, col=2)

    # panel 3: board power
    for col, label, color in [
        ("agent_board_power", "Agent power", "#0891b2"),
        ("opp_board_power", "Opp power", "#f59e0b"),
    ]:
        if col in ep.columns:
            fig.add_trace(go.Scatter(x=x, y=ep[col], mode="lines+markers",
                                     name=label, line=dict(color=color, width=2)),
                          row=2, col=1)

    # panel 4: base + leader HP
    for col, label, color in [
        ("agent_base_hp", "Agent base", "#3b82f6"),
        ("opp_base_hp", "Opp base", "#dc2626"),
        ("agent_leader_hp", "Agent leader", "#6366f1"),
        ("opp_leader_hp", "Opp leader", "#e11d48"),
    ]:
        if col in ep.columns:
            fig.add_trace(go.Scatter(x=x, y=ep[col], mode="lines+markers",
                                     name=label, line=dict(color=color, width=2)),
                          row=2, col=2)

    fig.update_layout(height=650, title_text=f"Episode {episode} — Turn‑by‑Turn",
                      legend=dict(orientation="h", y=1.12))
    fig.update_xaxes(title_text="Turn", dtick=1, row=2, col=1)
    fig.update_xaxes(title_text="Turn", dtick=1, row=2, col=2)
    return fig


def plot_last_turn_panel(turn_df: pd.DataFrame, window: int = 10) -> go.Figure:
    """End‑of‑game snapshot across training."""
    last_idx = turn_df.groupby("episode")["turn_number"].idxmax()
    last = turn_df.loc[last_idx].sort_values("training_order")
    last = last.groupby("episode", as_index=False).last()

    metrics = ["agent_base_hp", "opp_base_hp", "agent_unit_count", "opp_unit_count",
               "agent_board_power", "opp_board_power"]
    cols = 3
    rows = 2
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[TURN_METRICS[m][0] for m in metrics],
                        vertical_spacing=0.12)
    x = last["training_order"]
    for i, m in enumerate(metrics):
        if m not in last.columns:
            continue
        r, c = i // cols + 1, i % cols + 1
        _, color = TURN_METRICS.get(m, (m, "#888"))
        fig.add_trace(go.Scatter(x=x, y=last[m], mode="markers",
                                 marker=dict(size=3, color=color), opacity=0.35,
                                 showlegend=False), row=r, col=c)
        fig.add_trace(go.Scatter(x=x, y=rolling_mean(last[m], window), mode="lines",
                                 line=dict(color=color, width=2), showlegend=False),
                      row=r, col=c)

    fig.update_layout(height=600, title_text="End‑of‑Game Metrics Across Training")
    return fig


# ── summary stats ────────────────────────────────────────────────────

def summary_stats(df: pd.DataFrame) -> dict:
    return {
        "n_episodes":       len(df),
        "win_rate":         float(df["agent_win"].mean()),
        "avg_agent_reward": float(df["agent_rewards"].mean()),
        "avg_reward_margin":float(df["reward_margin"].mean()),
        "avg_steps":        float(df["steps"].mean()),
        "avg_turns":        float(df["agent_turns"].mean()),
        "best_margin_ep":   int(df.loc[df["reward_margin"].idxmax(), "episode"]),
        "best_margin":      float(df["reward_margin"].max()),
        "worst_margin_ep":  int(df.loc[df["reward_margin"].idxmin(), "episode"]),
        "worst_margin":     float(df["reward_margin"].min()),
    }
