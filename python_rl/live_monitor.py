"""
live_monitor.py — Real‑time training dashboard via Dash.

Launch while train.py is running:
    python live_monitor.py --run train_run11 --port 8051

It reads episode_summaries.csv every 5 seconds and updates plots live.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visu import find_run_dir, rolling_mean


def build_figure(csv_path: Path) -> go.Figure:
    """Build a 4‑panel training dashboard from the current CSV state."""
    if not csv_path.exists():
        return go.Figure()

    df = pd.read_csv(csv_path)
    if df.empty:
        return go.Figure()

    df["training_order"] = range(1, len(df) + 1)
    for c in ["agent_rewards", "opponent_rewards", "steps", "agent_turns",
              "agent_reward_per_turn", "cards_played"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["reward_margin"] = df["agent_rewards"] - df["opponent_rewards"]
    # win determined by winner column if available, else margin
    if "winner" in df.columns:
        w = df["winner"].astype(str).str.lower().str.strip()
        df["agent_win"] = w.isin({"agent", "111", "player1", "p1"})
        unresolved = w.isin({"none", "nan", "", "draw", "unresolved"})
        df.loc[unresolved, "agent_win"] = df.loc[unresolved, "reward_margin"] > 0
    else:
        df["agent_win"] = df["reward_margin"] > 0

    W = max(5, min(50, len(df) // 5))  # adaptive window
    x = df["training_order"]

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=("Win Rate", "Rewards", "Episode Length", "Reward Margin"),
        vertical_spacing=0.15, horizontal_spacing=0.08)

    # Panel 1: Win rate
    w = df["agent_win"].astype(float)
    fig.add_trace(go.Scatter(x=x, y=rolling_mean(w, W), mode="lines",
        name=f"Win rate (w={W})", line=dict(color="#22c55e", width=2)), row=1, col=1)

    # Panel 2: Rewards
    fig.add_trace(go.Scatter(x=x, y=df["agent_rewards"], mode="markers",
        name="Agent reward", marker=dict(size=2, color="#2563eb"), opacity=0.3,
        showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=rolling_mean(df["agent_rewards"], W), mode="lines",
        name=f"Agent (w={W})", line=dict(color="#1d4ed8", width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=rolling_mean(df["opponent_rewards"], W), mode="lines",
        name=f"Opponent (w={W})", line=dict(color="#dc2626", width=2, dash="dot")), row=1, col=2)

    # Panel 3: Steps
    fig.add_trace(go.Scatter(x=x, y=df["steps"], mode="markers",
        name="Steps", marker=dict(size=2, color="#0f766e"), opacity=0.3,
        showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=rolling_mean(df["steps"], W), mode="lines",
        name=f"Steps (w={W})", line=dict(color="#115e59", width=2)), row=2, col=1)

    # Panel 4: Reward margin
    fig.add_trace(go.Scatter(x=x, y=df["reward_margin"], mode="markers",
        name="Margin", marker=dict(size=2, color="#f97316"), opacity=0.25,
        showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=x, y=rolling_mean(df["reward_margin"], W), mode="lines",
        name=f"Margin (w={W})", line=dict(color="#ea580c", width=2)), row=2, col=2)
    fig.add_hline(y=0, line=dict(color="#94a3b8", width=1, dash="dot"), row=2, col=2)

    last = len(df)
    info_text = (
        f"Episodes: {last}  |  "
        f"Win rate: {df['agent_win'].mean():.1%}  |  "
        f"Avg reward: {df['agent_rewards'].mean():.2f}  |  "
        f"Avg margin: {df['reward_margin'].mean():+.2f}  |  "
        f"Avg steps: {df['steps'].mean():.0f}"
    )

    fig.update_layout(height=650, title_text=f"Live Training — {info_text}",
                      template="plotly_white", margin=dict(t=80))
    return fig


def main():
    parser = argparse.ArgumentParser(description="Live training monitor")
    parser.add_argument("--run", default="train", help="Run directory name under runs/")
    parser.add_argument("--port", type=int, default=8051)
    parser.add_argument("--interval", type=int, default=5, help="Refresh interval in seconds")
    args = parser.parse_args()

    run_dir = find_run_dir(args.run)
    csv_path = run_dir / "episode_summaries.csv"
    print(f"Watching {csv_path} — refresh every {args.interval}s")
    print(f"Dashboard at http://127.0.0.1:{args.port}")

    app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
    app.title = f"Live — {run_dir.name}"

    app.layout = dbc.Container([
        html.H3(f"Live Training Monitor — {run_dir.name}", className="mt-3"),
        html.P(f"Reading {csv_path}", className="text-muted small"),
        dcc.Interval(id="interval", interval=args.interval * 1000),
        dcc.Graph(id="live-graph", config={"responsive": True}),
    ], fluid=True)

    @app.callback(Output("live-graph", "figure"), Input("interval", "n_intervals"))
    def refresh(_n):
        return build_figure(csv_path)

    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
