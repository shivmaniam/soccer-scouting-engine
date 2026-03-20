"""
app/streamlit_app.py
--------------------
Standalone Streamlit UI — no API server required.
Loads embeddings and features directly from disk.

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make sure src/ is importable when launched from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embed import load_embeddings
from src.features import FEATURE_COLUMNS, load_features
from src.search import SimilarityIndex

st.set_page_config(
    page_title="Soccer Scouting Engine",
    page_icon="⚽",
    layout="wide",
)

# ── cached data loading ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading player index …")
def get_index() -> SimilarityIndex:
    return SimilarityIndex.from_disk()


@st.cache_resource(show_spinner="Loading feature data …")
def get_features() -> pd.DataFrame:
    return load_features()


@st.cache_data(show_spinner="Computing UMAP projection …")
def get_umap_coords() -> pd.DataFrame:
    import umap as umap_lib

    emb = load_embeddings()
    emb_cols = [c for c in emb.columns if c.startswith("emb_")]
    X = emb[emb_cols].values.astype(np.float32)

    # With small datasets n_neighbors must be < n_samples
    n_neighbors = min(15, len(X) - 1)
    xy = umap_lib.UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=42).fit_transform(X)

    result = emb[["player_name", "team"]].copy()
    result["x"] = xy[:, 0]
    result["y"] = xy[:, 1]
    return result


# ── chart builders ─────────────────────────────────────────────────────────────

# Subset of the 30 features shown on the radar
_RADAR_COLS = [
    "goals_p90", "shots_p90", "xg_p90",
    "key_passes_p90", "dribbles_completed_p90", "progressive_carries_p90",
    "pass_completion_pct", "progressive_passes_p90",
    "tackles_won_p90", "pressures_p90", "interceptions_p90",
]
_RADAR_LABELS = [
    "Goals", "Shots", "xG",
    "Key Passes", "Dribbles", "Prog Carries",
    "Pass Cmp%", "Prog Passes",
    "Tackles Won", "Pressures", "Interceptions",
]


def _radar(
    features: pd.DataFrame,
    pid_a: int,
    name_a: str,
    pid_b: int,
    name_b: str,
) -> go.Figure:
    """Radar chart comparing two players, values scaled 0-100 vs. all players."""
    available = [c for c in _RADAR_COLS if c in features.columns]
    labels = [_RADAR_LABELS[_RADAR_COLS.index(c)] for c in available]

    # Scale each feature to 0-100 percentile rank across all players
    ranked = features[available].rank(pct=True) * 100

    def _vals(pid: int) -> list[float]:
        if pid in ranked.index:
            return ranked.loc[pid, available].tolist()
        return [0.0] * len(available)

    theta = labels + [labels[0]]  # close the polygon

    fig = go.Figure()
    for pid, name, color in [(pid_a, name_a, "#00b4d8"), (pid_b, name_b, "#f77f00")]:
        vals = _vals(pid)
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=theta,
            fill="toself",
            name=name,
            line_color=color,
            opacity=0.65,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%")),
        legend=dict(orientation="h", y=-0.1),
        margin=dict(t=30, b=40, l=40, r=40),
        height=380,
    )
    return fig


def _umap_scatter(
    umap_df: pd.DataFrame,
    highlight_ids: list[int],
    query_id: int,
) -> go.Figure:
    """UMAP scatter with the query player and its neighbours highlighted."""
    neighbour_ids = [pid for pid in highlight_ids if pid != query_id]

    fig = go.Figure()

    # All players — grey background dots
    fig.add_trace(go.Scatter(
        x=umap_df["x"], y=umap_df["y"],
        mode="markers",
        marker=dict(size=5, color="#ced4da", opacity=0.6),
        text=umap_df["player_name"] + "<br>" + umap_df["team"],
        hoverinfo="text",
        name="All players",
        showlegend=False,
    ))

    # Neighbours — orange stars
    nb = umap_df.loc[umap_df.index.isin(neighbour_ids)]
    if not nb.empty:
        fig.add_trace(go.Scatter(
            x=nb["x"], y=nb["y"],
            mode="markers",
            marker=dict(size=10, color="#f77f00", symbol="star"),
            text=nb["player_name"] + "<br>" + nb["team"],
            hoverinfo="text",
            name="Similar players",
        ))

    # Query player — blue diamond
    if query_id in umap_df.index:
        qrow = umap_df.loc[query_id]
        fig.add_trace(go.Scatter(
            x=[qrow["x"]], y=[qrow["y"]],
            mode="markers+text",
            marker=dict(size=14, color="#00b4d8", symbol="diamond"),
            text=[qrow["player_name"]],
            textposition="top center",
            hoverinfo="text",
            name="Query player",
        ))

    fig.update_layout(
        title="Player Embedding Space (UMAP)",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        legend=dict(orientation="h", y=-0.05),
        margin=dict(t=40, b=10, l=10, r=10),
        height=460,
    )
    return fig


# ── main UI ───────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("Soccer Scouting Similarity Engine")
    st.caption("Find players across global leagues who play most like any given player.")

    # ── load data ──────────────────────────────────────────────────────────────
    try:
        index = get_index()
        features = get_features()
    except FileNotFoundError as exc:
        st.error(
            f"**Data not found.** Run the pipeline first:\n\n"
            f"```\nmake pipeline COMP_ID=11 SEASON_ID=90 MAX_MATCHES=10\n```\n\n"
            f"Details: `{exc}`"
        )
        return

    emb = index._embeddings
    all_names = sorted(emb["player_name"].dropna().unique().tolist())

    # ── sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Search")
        selected_name = st.selectbox("Player", all_names, index=0)
        top_k = st.slider("Similar players", min_value=3, max_value=20, value=10)
        show_umap = st.checkbox("Show embedding space (UMAP)", value=True)

    # ── resolve player ─────────────────────────────────────────────────────────
    name_lower = selected_name.lower()
    query_id = index._name_to_id.get(name_lower)
    if query_id is None:
        st.warning(f"'{selected_name}' not found in the index.")
        return

    query_row = emb.loc[query_id]

    # ── find similar ───────────────────────────────────────────────────────────
    with st.spinner("Searching …"):
        results = index.find_similar(query_id, top_k=top_k)

    # ── header ─────────────────────────────────────────────────────────────────
    st.subheader(f"Players most similar to {selected_name}")
    st.caption(f"{query_row['team']} · {query_row['total_minutes']:.0f} minutes played")

    # ── main columns ───────────────────────────────────────────────────────────
    col_table, col_radar = st.columns([1, 1], gap="large")

    with col_table:
        st.markdown("**Similarity results**")
        display = results[["player_name", "team", "total_minutes", "distance"]].copy()
        display["total_minutes"] = display["total_minutes"].astype(int)
        display["distance"] = display["distance"].round(3)
        display.index = range(1, len(display) + 1)
        st.dataframe(
            display.rename(columns={
                "player_name": "Player",
                "team": "Team",
                "total_minutes": "Mins",
                "distance": "Distance ↓",
            }),
            width="stretch",
        )

    with col_radar:
        if not results.empty:
            top_match_id = int(results.iloc[0]["player_id"])
            top_match_name = results.iloc[0]["player_name"]
            st.markdown(f"**Style comparison** vs. closest match")
            st.plotly_chart(
                _radar(features, query_id, selected_name, top_match_id, top_match_name),
                width="stretch",
            )

    # ── UMAP ───────────────────────────────────────────────────────────────────
    if show_umap:
        st.divider()
        try:
            umap_df = get_umap_coords()
            highlight_ids = [query_id] + results["player_id"].tolist()
            st.plotly_chart(
                _umap_scatter(umap_df, highlight_ids, query_id),
                width="stretch",
            )
        except Exception as exc:  # noqa: BLE001
            st.info(f"UMAP unavailable (`pip install umap-learn`). {exc}")

    # ── raw stats expander ─────────────────────────────────────────────────────
    with st.expander("Raw per-90 stats for all results"):
        pids = [query_id] + results["player_id"].tolist()
        names = [selected_name] + results["player_name"].tolist()
        available_cols = [c for c in FEATURE_COLUMNS if c in features.columns]
        rows = []
        for pid, name in zip(pids, names):
            if pid in features.index:
                row = {"Player": name}
                row.update(features.loc[pid, available_cols].apply(pd.to_numeric, errors="coerce").round(2).to_dict())
                rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("Player"), width="stretch")


if __name__ == "__main__":
    main()
