"""
turn_dashboard.py — Dash app for interactive turn‑by‑turn analysis.

Usage:
    python turn_dashboard.py [--run RUN_NAME] [--port PORT]

Then open http://127.0.0.1:8050 (or the specified port).
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# Make sure visu.py is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from visu import (
    find_run_dir, load_episode_summaries, load_turn_summaries,
    TURN_METRICS, rolling_mean, summary_stats,
)


# ── layout helpers ───────────────────────────────────────────────────

def _metric_card(label: str, value: str, color: str = "#6366f1") -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.H6(label, className="card-subtitle text-muted", style={"fontSize": "0.8rem"}),
            html.H4(value, className="card-title", style={"color": color, "fontWeight": "600"}),
        ]),
        className="shadow-sm",
    )


def _make_graph(fig: go.Figure) -> dcc.Graph:
    return dcc.Graph(figure=fig, config={"displayModeBar": True, "responsive": True})


# ── app factory ──────────────────────────────────────────────────────

def create_app(run_dir: Path) -> Dash:
    ep_df = load_episode_summaries(run_dir)
    turn_df = load_turn_summaries(run_dir, ep_df)
    stats = summary_stats(ep_df)

    turns_available = sorted(turn_df["turn_number"].dropna().unique().astype(int))
    episodes_available = sorted(ep_df["episode"].dropna().unique().astype(int))

    # Pre‑select middle turn and last episode
    default_turn = turns_available[len(turns_available) // 2] if turns_available else 1
    default_episode = episodes_available[-1] if episodes_available else 1

    app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
    app.title = f"SWU RL — Turn Dashboard ({run_dir.name})"

    # ── layout ───────────────────────────────────────────────────
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col(html.H3(f"Turn Dashboard — {run_dir.name}", className="mt-3 mb-0"), width=8),
            dbc.Col(html.P(f"{stats['n_episodes']} episodes · turns {turns_available[0]}–{turns_available[-1]}",
                           className="text-muted mt-3 mb-0 text-end"), width=4),
        ]),
        html.Hr(),

        # Summary cards
        dbc.Row([
            dbc.Col(_metric_card("Win Rate", f"{stats['win_rate']:.1%}", "#22c55e"), width=2),
            dbc.Col(_metric_card("Avg Reward", f"{stats['avg_agent_reward']:.2f}", "#2563eb"), width=2),
            dbc.Col(_metric_card("Avg Margin", f"{stats['avg_reward_margin']:+.2f}", "#f97316"), width=2),
            dbc.Col(_metric_card("Avg Steps", f"{stats['avg_steps']:.0f}", "#0f766e"), width=2),
            dbc.Col(_metric_card("Avg Turns", f"{stats['avg_turns']:.1f}", "#14b8a6"), width=2),
            dbc.Col(_metric_card("Best Margin", f"{stats['best_margin']:+.2f}", "#7c3aed"), width=2),
        ], className="mb-3"),

        # Turn selector
        dbc.Row([
            dbc.Col([
                html.Label("Turn number", className="fw-bold"),
                dcc.Slider(
                    id="turn-slider",
                    min=turns_available[0], max=turns_available[-1],
                    step=1, value=default_turn,
                    marks={t: str(t) for t in turns_available if t % 2 == 0 or t == turns_available[0] or t == turns_available[-1]},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ], width=12),
        ], className="mb-3"),

        # Turn dashboard (multi‑panel)
        dbc.Row([dbc.Col(_make_graph(go.Figure()), id="turn-dashboard-graph", width=12)]),

        html.Hr(),

        # Episode selector + single‑episode breakdown
        dbc.Row([
            dbc.Col([
                html.Label("Episode", className="fw-bold"),
                dcc.Dropdown(
                    id="episode-dropdown",
                    options=[{"label": f"Episode {e}", "value": e} for e in episodes_available],
                    value=default_episode, clearable=False,
                ),
            ], width=4),
        ], className="mb-3"),

        dbc.Row([dbc.Col(_make_graph(go.Figure()), id="episode-turn-graph", width=12)]),

        html.Hr(),
        html.P("visu.py · SWU RL postmortem · turn_summaries.csv", className="text-muted small text-center"),
    ], fluid=True)

    # ── callbacks ─────────────────────────────────────────────────

    @app.callback(
        Output("turn-dashboard-graph", "figure"),
        Input("turn-slider", "value"),
    )
    def update_turn_dashboard(turn: int):
        metrics = [
            "agent_unit_count", "opp_unit_count", "agent_exhausted_count",
            "agent_ready_resources", "agent_hand_count",
            "agent_base_hp", "opp_base_hp",
            "agent_board_power", "opp_board_power",
        ]
        sub = turn_df[turn_df["turn_number"] == turn].sort_values("training_order")
        if sub.empty:
            return go.Figure()

        from plotly.subplots import make_subplots
        n = len(metrics)
        cols = 3
        rows = (n + cols - 1) // cols
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
            fig.add_trace(go.Scatter(x=x, y=rolling_mean(sub[m], 10), mode="lines",
                                     name=f"{label} avg", line=dict(color=color, width=2),
                                     showlegend=False), row=r, col=c)
        fig.update_layout(height=280 * rows, title_text=f"Turn {turn} — Dashboard",
                          template="plotly_white")
        return fig

    @app.callback(
        Output("episode-turn-graph", "figure"),
        Input("episode-dropdown", "value"),
    )
    def update_episode_turns(ep: int):
        from visu import plot_episode_turns
        try:
            return plot_episode_turns(turn_df, ep)
        except ValueError:
            return go.Figure()

    return app


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Turn‑by‑turn Dash dashboard for SWU RL runs")
    parser.add_argument("--run", default="train_run8", help="Run directory name under runs/")
    parser.add_argument("--port", type=int, default=8050, help="Dash server port")
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode")
    args = parser.parse_args()

    run_dir = find_run_dir(args.run)
    print(f"Loading data from {run_dir} ...")
    app = create_app(run_dir)
    print(f"Starting dashboard on http://127.0.0.1:{args.port}")
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()
