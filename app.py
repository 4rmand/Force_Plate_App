# Import librairies 
import dash
from dash import Dash, html, dcc, Input, Output, State, dash_table, no_update
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.load_force_data import load_force_data, load_all_files_for_athlete
from utils.plot_force_curve import plot_force_curve
import os
from utils.cmj_extract import extract_cmj, compute_acc_vel, detect_cmj_phases, compute_phase_metrics
import itertools
import plotly
import matplotlib
matplotlib.use("Agg")  # Non-GUI backend (renders in memory)
import matplotlib.pyplot as plt
import json
import plotly.colors
import re

# --- Initialisation de l'app ---
app = dash.Dash(__name__, title="Analyse CMJ – Plateforme de force")
app.config.suppress_callback_exceptions = True

# --- Chemin des données ---
default_path = "/Users/Armand/Desktop/Python/VALD"

# --- Layout ---
app.layout = html.Div([
    html.H2("Analyse CMJ – Plateforme de force"),

    html.Div([
        html.Label("Chemin des données :"),
        dcc.Input(id="data_path", type="text", value=default_path, style={"width": "80%"}),
        html.Button("Charger les fichiers", id="load_btn", n_clicks=0)
    ], style={"marginBottom": "20px"}),

    html.Div([
        html.Label("Sélectionner un athlète :"),
        dcc.Dropdown(id="athlete_selected", multi=True, style={"width": "80%"})
    ], style={"marginBottom": "10px"}),


    html.Button("Detect Jumps", id="detect_button", n_clicks=0),

    # --- NEW COMPACT SELECTION AREA ---
    html.Div([
        html.Label("Select session date(s):"),
        dcc.Dropdown(id="date-dropdown", multi=True, style={"width": "80%"}),
    ], style={"marginBottom": "10px"}),

    html.Div([
        html.Label("Select CMJ(s):"),
        dcc.Dropdown(id="trial_dropdown", multi=True, style={"width": "80%"}),
    ], style={"marginBottom": "20px"}),

    
    html.Div([
        dcc.Checklist(id="rel_time",
                    options=[{"label": " Relative time (0-100 %)", "value": "t"}],
                    value=[], inline=True),
        dcc.Checklist(id="rel_force",
                    options=[{"label": " Relative force (% BW)", "value": "f"}],
                    value=[], inline=True),
        dcc.Checklist(id="show_vel",
                    options=[{"label": " Show velocity", "value": "on"}],
                    value=[], inline=True),
        dcc.Checklist(id="show_phases",
                    options=[{"label": " Show phase markers & backgrounds", "value": "on"}],
                    value=[], inline=True),
        dcc.Checklist(id="leg_toggle",
                    options=[{'label': ' Show Left & Right Leg', 'value': 'show_legs'}],
                    value=[], inline=True)
    ], style={"marginBottom": "10px"}),

    # NEW: Phase selection checkboxes (only visible when "show phases" is ticked)
    html.Div([
        html.Label("Select phases to display:", style={"fontWeight": "bold"}),
        dcc.Checklist(
            id="phase_selector",
            options=[
                {"label": " Unweighting", "value": "unweighting"},
                {"label": " Braking", "value": "braking"},
                {"label": " Propulsive", "value": "propulsive"},
                {"label": " Flight", "value": "flight"}
            ],
            value=["unweighting", "braking", "propulsive", "flight"],  # All selected by default
            inline=True
        )
    ], id="phase_selector_container", style={"marginBottom": "10px", "marginLeft": "20px"}),

    dcc.Graph(id="force_plot", style={"height": "60vh"}),


    html.Div([
        html.H4("Overall Jump Metrics", style={"textAlign": "center", "color": "blue", "marginTop": "20px"}),
        dash_table.DataTable(
            id="overall_table",
            style_table={"overflowX": "auto", "width": "100%"},
            style_cell={"textAlign": "center", "fontSize": 12, "padding": "4px"},
            style_header={"fontWeight": "bold", "backgroundColor": "#e3f2fd"},
            export_format="csv"
        )
    ], style={"width": "100%", "marginTop": "20px"}),


    ### Table
    html.Div([
    html.Div([
        html.H4("Unweighting", style={"textAlign": "center", "color": "red"}),
        dash_table.DataTable(
            id="unw_table",
            style_table={"overflowX": "auto", "width": "100%"},
            style_cell={"textAlign": "center", "fontSize": 12, "padding": "4px"},
            export_format="csv"
        )
    ], style={"width": "33%", "display": "inline-block", "verticalAlign": "top"}),

    html.Div([
        html.H4("Braking", style={"textAlign": "center", "color": "orange"}),
        dash_table.DataTable(
            id="brk_table",
            style_table={"overflowX": "auto", "width": "100%"},
            style_cell={"textAlign": "center", "fontSize": 12, "padding": "4px"},
            export_format="csv"
        )
    ], style={"width": "33%", "display": "inline-block", "verticalAlign": "top"}),

    html.Div([
        html.H4("Propulsive", style={"textAlign": "center", "color": "green"}),
        dash_table.DataTable(
            id="prop_table",
            style_table={"overflowX": "auto", "width": "100%"},
            style_cell={"textAlign": "center", "fontSize": 12, "padding": "4px"},
            export_format="csv"
        )
    ], style={"width": "33%", "display": "inline-block", "verticalAlign": "top"}),
]),


    dcc.Store(id="phase_metrics_store", data=[]),
    dcc.Store(id='detected_jumps_store'),


    html.Div(id="summary_table")
])

# -------------------------------------------------
# 1.  Athlete + file picker
# -------------------------------------------------
@app.callback(
    Output("athlete_selected", "options"),
    Input("load_btn", "n_clicks"),
    State("data_path", "value")
)
def list_athletes(n, path):
    if not os.path.exists(path):
        return []
    files = [f for f in os.listdir(path) if f.lower().endswith(".csv")]
    if not files:
        return []
    athletes = sorted({f.split("-")[0].strip() for f in files})
    return [{"label": a, "value": a} for a in athletes]
    
# ------------------------------------------------------------------
# 1.2  Fill file dropdown when athlete is picked
# ------------------------------------------------------------------
@app.callback(
    Output("file_list", "options"),
    Output("file_list", "value"),
    Input("athlete_selected", "value"),
    State("data_path", "value")
)
def fill_file_dropdown(athlete, path):
    if not athlete:
        return [], []
    all_data = load_all_files_for_athlete(path, athlete)
    options = [{"label": f, "value": f} for f in sorted(all_data.keys())]
    return options, [options[0]["value"]] if options else []


# ------------------------------------------------------------------
#  Detect-jumps button → fill date & trial drop-downs
# ------------------------------------------------------------------
@app.callback(
    Output("date-dropdown", "options"),
    Output("date-dropdown", "value"),  # ← ADD THIS
    Output("trial_dropdown", "options"),
    Output("trial_dropdown", "value"),  # ← ADD THIS
    Output("detected_jumps_store", "data"),
    Input("detect_button", "n_clicks"),
    Input("athlete_selected", "value"),
    Input("date-dropdown", "value"),
    State("data_path", "value"),
    State("detected_jumps_store", "data"),
    prevent_initial_call=True
)
def manage_jump_detection_and_filter(n_clicks, athletes, selected_dates, path, detected_jumps):
    """
    Unified callback:
    - Detect jumps when Detect button clicked or athlete changes.
    - Filter CMJs when date(s) are selected.
    """
    triggered_id = dash.callback_context.triggered_id

    # --- CASE 1: Detect button clicked or athlete changed ---
    if triggered_id in ["detect_button", "athlete_selected"]:
        if not athletes:
            raise PreventUpdate

        if isinstance(athletes, str):
            athletes = [athletes]

        detected_jumps = {}
        for athlete in athletes:
            for fname in os.listdir(path):
                if not fname.lower().endswith(".csv") or athlete not in fname:
                    continue

                df, meta = load_force_data(os.path.join(path, fname))
                jumps = extract_cmj(df, debug=True)

                date_match = re.search(r"(\d{4}\.\d{2}\.\d{2})", fname)
                date_str = date_match.group(1) if date_match else "Unknown"

                for j_idx, (df_jump, meta) in enumerate(jumps):
                    ft_ms = int(meta.get("flight_time", 0) * 1000)
                    key = f"{athlete}_{fname}_jump{j_idx+1}"
                    detected_jumps[key] = {
                        "file": fname,
                        "athlete": athlete,
                        "date": date_str,
                        "jump_index": j_idx,
                        "metrics": meta,
                        "data": df_jump.to_dict("records"),
                        "label": f"{athlete} – {date_str} – CMJ #{j_idx+1} – flight {ft_ms} ms"
                    }

        # Dropdowns
        date_opts = [{"label": d, "value": d}
                     for d in sorted({j["date"] for j in detected_jumps.values()})]
        trial_opts = [{"label": j["label"], "value": k}
                      for k, j in detected_jumps.items()]

        # ← CLEAR THE SELECTIONS when detecting new jumps
        return date_opts, None, trial_opts, None, detected_jumps

    # --- CASE 2: Date(s) selected ---
    elif triggered_id == "date-dropdown":
        if not selected_dates or not detected_jumps:
            raise PreventUpdate

        if isinstance(selected_dates, str):
            selected_dates = [selected_dates]

        filtered_trials = [
            {"label": j["label"], "value": k}
            for k, j in detected_jumps.items()
            if j["date"] in selected_dates
        ]

        date_opts = [{"label": d, "value": d}
                     for d in sorted({j["date"] for j in detected_jumps.values()})]

        # ← Keep the date selection, clear trial selection when dates change
        return date_opts, selected_dates, filtered_trials, None, detected_jumps

    else:
        raise PreventUpdate



# ------------------------------------------------------------------
# 3.  COMBINED Plot - handles all visualization options
# ------------------------------------------------------------------
@app.callback(
    Output("force_plot", "figure"),
    Output("phase_metrics_store", "data"),
    Input("trial_dropdown", "value"),
    Input("rel_time", "value"),
    Input("rel_force", "value"),
    Input("show_vel", "value"),
    Input("show_phases", "value"),
    Input("phase_selector", "value"),
    Input("leg_toggle", "value"),
    State("detected_jumps_store", "data"),
    prevent_initial_call=True
)
def update_plot(selected_jumps, rel_time, rel_force, show_vel, show_phases, selected_phases, show_legs, detected_jumps):
    if not detected_jumps or not selected_jumps:
        raise dash.exceptions.PreventUpdate

    fig = go.Figure()
    colours = itertools.cycle(plotly.colors.qualitative.Plotly)

    # ---- axis labels ----
    if rel_time and rel_force:
        x_lab, y_lab = "Time (% jump)", "Force (% BW)"
    elif rel_time:
        x_lab, y_lab = "Time (% jump)", "Force (N)"
    elif rel_force:
        x_lab, y_lab = "Time (s)", "Force (% BW)"
    else:
        x_lab, y_lab = "Time (s)", "Force (N)"

    # Storage for metrics from all selected jumps
    unw_metrics = []
    brk_metrics = []
    prop_metrics = []
    overall_metrics = []  # ADD THIS

    for jump_key in selected_jumps:
        jump_data = detected_jumps.get(jump_key)
        if not jump_data:
            continue

        # ---- rebuild DataFrame ----
        df_jump = pd.DataFrame(jump_data["data"])
        mj = jump_data["metrics"]
        fname = jump_data["file"]
        j_idx = jump_data["jump_index"]
        bw = mj["body_weight"]

        # Compute velocity
        fs = int(1 / np.diff(df_jump["Time"]).mean())
        df_vel = compute_acc_vel(df_jump, fs, bw)

        # Detect phases
        phases = detect_cmj_phases(df_vel, fs, bw, debug=False)
        
        # ===== COMPUTE METRICS (on full data BEFORE cropping) =====
        if phases and len(phases) > 0:
            try:
                phase_metrics = compute_phase_metrics(df_vel, phases, bw, fs)
                
                jump_label = f"{fname} CMJ#{j_idx+1}"
                
                if phase_metrics.get("unw"):
                    unw_row = {"Jump": jump_label}
                    unw_row.update(phase_metrics["unw"])
                    unw_metrics.append(unw_row)
                
                if phase_metrics.get("brk"):
                    brk_row = {"Jump": jump_label}
                    brk_row.update(phase_metrics["brk"])
                    brk_metrics.append(brk_row)
                
                if phase_metrics.get("prop"):
                    prop_row = {"Jump": jump_label}
                    prop_row.update(phase_metrics["prop"])
                    prop_metrics.append(prop_row)
                
                if phase_metrics.get("overall"):
                    overall_row = {"Jump": jump_label}
                    overall_row.update(phase_metrics["overall"])
                    overall_metrics.append(overall_row)
            except Exception as e:
                print(f"[ERROR] Metrics failed for {fname} CMJ#{j_idx+1}: {e}")
        
        # ---- NOW crop for visualization ----
        if phases and 'unweight_start' in phases:
            unweight_time = phases['unweight_start']
            crop_start_time = max(0, unweight_time - 0.3)
            df_vel = df_vel[df_vel['Time'] >= crop_start_time].copy()
            df_vel = df_vel.reset_index(drop=True)
        
        # ---- NORMALIZE TIME: 0% = unweight start, 100% = takeoff ----
        if 't' in rel_time and phases and 'unweight_start' in phases and 'takeoff' in phases:
            unweight_time = phases['unweight_start']
            takeoff_time = phases['takeoff']
            jump_duration = takeoff_time - unweight_time
            
            if jump_duration > 0:
                # Create normalized time: 0% at unweight start, 100% at takeoff
                df_vel['TimePct'] = ((df_vel['Time'] - unweight_time) / jump_duration) * 100
                
                # Update phase times to percentages
                phases_normalized = {}
                for phase_name, phase_time in phases.items():
                    phases_normalized[phase_name] = ((phase_time - unweight_time) / jump_duration) * 100
                phases = phases_normalized
        else:
            df_vel['TimePct'] = df_vel['Time']
        
        # ---- Force normalization ----
        if 'f' in rel_force:
            df_vel['ForcePctBW'] = (df_vel['Z Total'] / bw) * 100
        
        # ---- pick axis data ----
        if 't' in rel_time and 'f' in rel_force:
            x, y = df_vel['TimePct'], df_vel['ForcePctBW']
        elif 't' in rel_time:
            x, y = df_vel['TimePct'], df_vel['Z Total']
        elif 'f' in rel_force:
            x, y = df_vel['Time'], df_vel['ForcePctBW']
        else:
            x, y = df_vel['Time'], df_vel['Z Total']

        # ---- base trace ----
        col = next(colours)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
                                 name=f"{fname} CMJ#{j_idx+1}",
                                 line=dict(width=2, color=col)))

        # ---- PHASE VISUALS (with selection) ----
        if 'on' in show_phases and phases and len(phases) > 0:
            use_time_norm = 't' in rel_time
            
            # Define phase regions
            phase_regions = []
            
            if 'unweighting' in selected_phases and 'unweight_start' in phases and 'braking_start' in phases:
                phase_regions.append({
                    'x0': phases['unweight_start'],
                    'x1': phases['braking_start'],
                    'color': 'purple',
                    'name': 'Unweighting'
                })
            
            if 'braking' in selected_phases and 'braking_start' in phases and 'propulsive_start' in phases:
                phase_regions.append({
                    'x0': phases['braking_start'],
                    'x1': phases['propulsive_start'],
                    'color': 'orange',
                    'name': 'Braking'
                })
            
            if 'propulsive' in selected_phases and 'propulsive_start' in phases and 'takeoff' in phases:
                phase_regions.append({
                    'x0': phases['propulsive_start'],
                    'x1': phases['takeoff'],
                    'color': 'green',
                    'name': 'Propulsive'
                })
            
            if 'flight' in selected_phases and 'takeoff' in phases and 'landing' in phases:
                phase_regions.append({
                    'x0': phases['takeoff'],
                    'x1': phases['landing'],
                    'color': 'lightblue',
                    'name': 'Flight'
                })
            
            # Draw selected phase regions
            for region in phase_regions:
                fig.add_vrect(
                    x0=region['x0'], x1=region['x1'],
                    fillcolor=region['color'], opacity=0.2, line_width=0,
                    annotation_text=region['name'], annotation_position="top left"
                )
            
            # Vertical markers (only show if relevant phase is selected)
            phase_markers = {}
            
            if 'unweighting' in selected_phases:
                if 'unweight_start' in phases:
                    phase_markers['unweight_start'] = ('purple', 'Unweight Start')
            
            if 'braking' in selected_phases:
                if 'braking_start' in phases:
                    phase_markers['braking_start'] = ('red', 'Min Force')
            
            if 'propulsive' in selected_phases:
                if 'propulsive_start' in phases:
                    phase_markers['propulsive_start'] = ('green', 'Propulsive Start')
                if 'peak_force' in phases:
                    phase_markers['peak_force'] = ('darkgreen', 'Peak Force')
            
            if 'flight' in selected_phases or 'propulsive' in selected_phases:
                if 'takeoff' in phases:
                    phase_markers['takeoff'] = ('blue', 'Takeoff')
            
            if 'flight' in selected_phases:
                if 'landing' in phases:
                    phase_markers['landing'] = ('navy', 'Landing')
            
            # Draw markers
            for phase_name, (color, label) in phase_markers.items():
                if phase_name in phases:
                    fig.add_vline(x=phases[phase_name], line_dash="dash", 
                                line_color=color, line_width=2)
                    fig.add_annotation(
                        x=phases[phase_name], y=1.02, xref="x", yref="paper",
                        text=label,
                        showarrow=False, font=dict(size=10, color=color), yanchor="bottom"
                    )

        # ---- left / right legs ----
        if 'show_legs' in show_legs and 'Z Left' in df_vel.columns and 'Z Right' in df_vel.columns:
            if 't' in rel_time and 'f' in rel_force:
                xL, yL = df_vel['TimePct'], (df_vel['Z Left'] / bw) * 100
                xR, yR = df_vel['TimePct'], (df_vel['Z Right'] / bw) * 100
            elif 't' in rel_time:
                xL, yL = df_vel['TimePct'], df_vel['Z Left']
                xR, yR = df_vel['TimePct'], df_vel['Z Right']
            elif 'f' in rel_force:
                xL, yL = df_vel['Time'], (df_vel['Z Left'] / bw) * 100
                xR, yR = df_vel['Time'], (df_vel['Z Right'] / bw) * 100
            else:
                xL, yL = df_vel['Time'], df_vel['Z Left']
                xR, yR = df_vel['Time'], df_vel['Z Right']

            fig.add_trace(go.Scatter(x=xL, y=yL, mode="lines",
                                     name=f"{fname} CMJ#{j_idx+1} – Left",
                                     line=dict(width=1.5, color="blue", dash="dot")))
            fig.add_trace(go.Scatter(x=xR, y=yR, mode="lines",
                                     name=f"{fname} CMJ#{j_idx+1} – Right",
                                     line=dict(width=1.5, color="red", dash="dot")))

        # ---- velocity on y2 ----
        if 'on' in show_vel:
            x_vel = df_vel['TimePct'] if 't' in rel_time else df_vel['Time']
            fig.add_trace(go.Scatter(x=x_vel, y=df_vel['Vel'],
                                     mode="lines", name=f"{fname} velocity",
                                     line=dict(width=2, color=col, dash="dot"),
                                     yaxis="y2"))

        # ---- body-weight horizontal line ----
        if 'f' not in rel_force:
            fig.add_hline(y=bw, line_dash="dash", line_color="grey",
                          annotation_text=f"BW = {bw:.1f} N",
                          annotation_position="right")

    # ---------- final layout ----------
    layout_kwargs = dict(template="plotly_white", xaxis_title=x_lab,
                         yaxis=dict(title=y_lab), hovermode="closest",
                         showlegend=True, legend=dict(x=0.01, y=0.99))
    if 'on' in show_vel:
        layout_kwargs["yaxis2"] = dict(title="Velocity (m/s)", overlaying="y", side="right")
    fig.update_layout(**layout_kwargs)

    return fig, {
    "unw": unw_metrics, 
    "brk": brk_metrics, 
    "prop": prop_metrics,
    "overall": overall_metrics  # ← ADD THIS
}

# ------------------------------------------------------------------
# 4.  Metric Tables
# ------------------------------------------------------------------

@app.callback(
    Output("unw_table", "data"),
    Output("brk_table", "data"),
    Output("prop_table", "data"),
    Output("overall_table", "data"),  # ADD THIS
    Input("phase_metrics_store", "data")
)
def update_phase_tables(metrics):
    if not metrics:
        return [], [], [], []  # UPDATE THIS
    return (
        metrics.get("unw", []), 
        metrics.get("brk", []), 
        metrics.get("prop", []),
        metrics.get("overall", [])  # ADD THIS
    )


# ------------------------------------------------------------------
#  Show/hide file dropdown
# ------------------------------------------------------------------
@app.callback(
    Output("file_container", "style"),
    Input("file_toggle", "n_clicks")
)
def toggle_file(n):
    return {"display": "block"} if (n or 0) % 2 else {"display": "none"}

# ------------------------------------------------------------------
#  Show/hide jump dropdown
# ------------------------------------------------------------------
@app.callback(
    Output("jump_container", "style"),
    Input("jump_toggle", "n_clicks")
)
def toggle_jump(n):
    return {"display": "block"} if (n or 0) % 2 else {"display": "none"}


# ------------------------------------------------------------------
# Show/hide phase selector based on "show_phases" checkbox
# ------------------------------------------------------------------
@app.callback(
    Output("phase_selector_container", "style"),
    Input("show_phases", "value")
)
def toggle_phase_selector(show_phases):
    if 'on' in show_phases:
        return {"marginBottom": "10px", "marginLeft": "20px", "display": "block"}
    else:
        return {"display": "none"}


if __name__ == "__main__":
    app.run(debug=True)