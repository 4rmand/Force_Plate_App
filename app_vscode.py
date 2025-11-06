# app_vscode.py
"""
Dash force-plate viewer that opens INSIDE VS-Code.
Run this file directly in VS-Code (Ctrl+F5 or ▶️).
"""
from __future__ import annotations
import os, threading, socket, time, webbrowser, tempfile, json
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, dash_table, ALL, Patch
from utils.load_force_data import load_all_files_for_athlete   # your old helpers
import itertools

# -------------------------------------------------
# 0.  CONFIG – change only here
# -------------------------------------------------
DATA_FOLDER = "/Users/Armand/Desktop/Python/VALD"   # ⇠ your folder
HOST = "127.0.0.1"
PORT = 0   # 0 = let OS pick free port

# -------------------------------------------------
# 1.  Tiny VS-Code helper
# -------------------------------------------------
def _get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

def open_in_vscode_panel(url: str):
    """
    VS-Code has a built-in web-view that is used for the Jupyter data-viewer.
    We piggy-back on it: create a dummy .html file that simply redirects to
    our Dash address, then ask VS-Code to open it.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    tmp.write(f"""
    <!doctype html>
    <html><head><meta http-equiv="refresh" content="0; url={url}"></head></html>
    """.encode())
    tmp.close()
    # The next line works because VS-Code registers itself as opener for .html
    # when the "Jupyter" extension is installed.
    webbrowser.open(f"vscode://file/{tmp.name}")

# -------------------------------------------------
# 2.  Dash app – identical philosophy, richer UI
# -------------------------------------------------
app = dash.Dash(__name__, title="Force-plate analyser")
server = app.server

app.layout = html.Div([
    html.H3("Force-plate CMJ analyser (VS-Code edition)"),
    html.Hr(),

    html.Div(style={"display": "flex", "gap": "20px"}, children=[
        html.Div(style={"width": "30%"}, children=[
            html.Label("Athlete filter (type or pick)"),
            dcc.Dropdown(id="athlete_dd", multi=True, placeholder="Start typing…"),
            html.Br(),
            html.Label("Trial selector"),
            html.Div(id="trial_checklist"),
            html.Br(),
            html.Label("Normalisation"),
            dcc.RadioItems(id="norm_radio",
                         options=[{"label": "Raw time", "value": "raw"},
                                 {"label": "0–100 % of jump", "value": "pct"}],
                         value="pct",
                         inline=True),
        ]),
        html.Div(style={"flex": 1}, children=[
            dcc.Graph(id="main_plot", style={"height": "70vh"}),
            dash_table.DataTable(id="summary_tbl",
                               style_table={"overflowX": "auto"},
                               style_cell={"textAlign": "center", "minWidth": "80px"})
        ])
    ])
])

# ---------- populate athlete list ----------
@app.callback(Output("athlete_dd", "options"),
             Input("athlete_dd", "search_value"))
def list_athletes(search):
    if not os.path.exists(DATA_FOLDER):
        return []
    files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(".csv")]
    athletes = sorted({f.split("-")[0].strip() for f in files})
    if search:
        athletes = [a for a in athletes if search.lower() in a.lower()]
    return [{"label": a, "value": a} for a in athletes]

# ---------- build trial checklist ----------
@app.callback(Output("trial_checklist", "children"),
             Input("athlete_dd", "value"))
def build_checklist(selected_athletes):
    if not selected_athletes:
        return html.Div("Select athlete(s) first")
    trials = []
    for ath in selected_athletes:
        try:
            for fname, (df, meta) in load_all_files_for_athlete(DATA_FOLDER, ath).items():
                trials.append({"label": f"{ath} – {fname}", "value": json.dumps([ath, fname])})
        except FileNotFoundError:
            continue
    return dcc.Checklist(id="trial_list", options=trials, value=[])

# ---------- main plotting ----------
@app.callback(Output("main_plot", "figure"),
             Output("summary_tbl", "data"),
             Input("trial_list", "value"),
             Input("norm_radio", "value"))
def plot_trials(trial_jsons, norm_mode):
    fig = go.Figure()
    summary = []
    colours = itertools.cycle(plotly.colors.qualitative.Plotly)
    for js in trial_jsons:
        ath, fname = json.loads(js)
        df, meta = load_all_files_for_athlete(DATA_FOLDER, ath)[fname]
        x = df["Time"].to_numpy()
        y = df["Z Total"].to_numpy()
        if norm_mode == "pct":
            # simple 0–100 % normalisation
            x = (x - x[0]) / (x[-1] - x[0]) * 100
        fig.add_trace(go.Scatter(x=x, y=y, name=f"{ath} – {fname}",
                                line=dict(width=2, color=next(colours))))
        summary.append({
            "Athlete": ath,
            "File": fname,
            "Date": meta.get("Recording Date", ""),
            "Points": len(df),
            "Fmax (N)": round(y.max(), 1),
            "Fmin (N)": round(y.min(), 1)
        })
    fig.update_layout(template="plotly_white",
                     xaxis_title="Time (% of jump)" if norm_mode == "pct" else "Time (s)",
                     yaxis_title="Force (N)")
    return fig, summary

# -------------------------------------------------
# 3.  Boot sequence: thread + VS-Code panel
# -------------------------------------------------
if __name__ == "__main__":
    PORT = _get_free_port()
    url = f"http://{HOST}:{PORT}"
    threading.Timer(1.5, lambda: open_in_vscode_panel(url)).start()
    app.run(host=HOST, port=PORT, debug=False)