import plotly.graph_objects as go
import pandas as pd

def plot_force_curve(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Time"],
        y=df["Force"],
        mode="lines",
        line=dict(color="#1E3D59"),
        name="Force"
    ))

    fig.update_layout(
        title="Courbe Force-Temps (CMJ)",
        xaxis_title="Temps (s)",
        yaxis_title="Force (N)",
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig
