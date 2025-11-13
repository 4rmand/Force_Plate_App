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
# At the top with your other imports
from utils.database import init_database, save_jump_to_db, get_all_athletes, get_all_jumps_dataframe


# Initialize database when app starts
init_database()

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

# ===== TABS START HERE =====
dcc.Tabs(id="main_tabs", value='tab-curve', children=[
    
    # ==================== TAB 1: CURVE ANALYSIS ====================
    dcc.Tab(label='📈 Curve Analysis', value='tab-curve', children=[
        html.Div([
            # Checkboxes for curve options
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
            ], style={"marginBottom": "10px", "marginTop": "20px"}),

            # Phase selection checkboxes
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
                    value=["unweighting", "braking", "propulsive", "flight"],
                    inline=True
                )
            ], id="phase_selector_container", style={"marginBottom": "10px", "marginLeft": "20px"}),

            # Force-time curve plot
            dcc.Graph(id="force_plot", style={"height": "60vh"}),

            # Overall metrics table
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

            # Phase tables
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
        ], style={"padding": "20px"})
    ]),
    
            # ==================== TAB 2: VERTICAL VECTOR ====================
            dcc.Tab(label='🚀 Vertical Vector', value='tab-vector', children=[
                html.Div([
                    html.H3("🏆 Athlete Performance Analysis", style={"textAlign": "center", "marginTop": "20px"}),
                    
                    # Show which athlete is being analyzed
                    html.Div(id="focus_athlete_display", style={"textAlign": "center", "marginBottom": "20px"}),
                    
                    # Comparison controls
                    html.Div([
                        html.Div([
                            html.Label("Select metric:"),
                            dcc.Dropdown(
                                id="compare_metric",
                                options=[
                                    {"label": "Jump Height (cm)", "value": "jump_height_cm"},
                                    {"label": "RSI-modified", "value": "rsi_modified"},
                                    {"label": "Peak Power (W/kg)", "value": "peak_power_rel"},
                                    {"label": "Mean Power (W/kg)", "value": "mean_power_rel"},
                                    {"label": "Takeoff Velocity (m/s)", "value": "takeoff_velocity"},
                                    {"label": "Contraction Time (s)", "value": "contraction_time"},
                                    {"label": "Peak Propulsive Force (% BW)", "value": "prop_max_force_pct"},
                                ],
                                value="jump_height_cm",
                                style={"width": "100%"}
                            ),
                        ], style={"width": "35%", "display": "inline-block", "paddingRight": "20px"}),
                        
                        html.Div([
                            html.Label("Compare to athlete (optional):"),
                            dcc.Dropdown(id="compare_athlete", placeholder="Select athlete to compare...", style={"width": "100%"}),
                        ], style={"width": "35%", "display": "inline-block", "paddingRight": "20px"}),
                        
                        html.Div([
                            html.Br(),
                            html.Button("Show Analysis", id="compare_btn", 
                                    style={"marginTop": "5px", "width": "100%"}),
                        ], style={"width": "20%", "display": "inline-block"}),
                    ], style={"marginBottom": "30px"}),
        
            
            # Time series plot with percentile bands
            dcc.Graph(id="comparison_plot", style={"height": "50vh"}),
            
            # Statistics table
            html.Div([
                html.H4("Population Statistics"),
                dash_table.DataTable(
                    id="population_stats_table",
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "center", "fontSize": 12},
                    style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"}
                )
            ], style={"marginTop": "20px"}),
            
            # Pizza plot
            html.Div([
                html.H4("Percentile Profile (Pizza Plot)", style={"textAlign": "center", "marginTop": "30px"}),
                dcc.Graph(id="pizza_plot", style={"height": "50vh"})
            ], style={"marginTop": "20px"})
            
        ], style={"padding": "20px", "backgroundColor": "#f9f9f9", "borderRadius": "10px"})
    ])
]),

# Storage components (keep these outside tabs)
dcc.Store(id="phase_metrics_store", data=[]),
dcc.Store(id='detected_jumps_store'),
html.Div(id="summary_table"),
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
    Output("date-dropdown", "value"),
    Output("trial_dropdown", "options"),
    Output("trial_dropdown", "value"),
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
                    
                    # ===== NEW: COMPUTE METRICS AND SAVE TO DATABASE =====
                    try:
                        fs = int(1 / np.diff(df_jump["Time"]).mean())
                        df_vel = compute_acc_vel(df_jump, fs, meta["body_weight"])
                        phases = detect_cmj_phases(df_vel, fs, meta["body_weight"], debug=False)
                        
                        if phases and len(phases) > 0:
                            phase_metrics = compute_phase_metrics(df_vel, phases, meta["body_weight"], fs)
                            
                            # Prepare metrics dict for database
                            metrics_for_db = {
                                'unw': phase_metrics.get('unw', {}),
                                'brk': phase_metrics.get('brk', {}),
                                'prop': phase_metrics.get('prop', {}),
                                'overall': phase_metrics.get('overall', {}),
                                'body_weight': meta["body_weight"],
                                'flight_time': meta.get("flight_time", 0)
                            }
                            
                            # Save to database
                            save_jump_to_db(
                                jump_id=key,
                                athlete_name=athlete,
                                session_date=date_str,
                                file_name=fname,
                                jump_number=j_idx + 1,
                                metrics_dict=metrics_for_db
                            )
                            print(f"✅ Saved {key} to database")
                    except Exception as e:
                        print(f"❌ Failed to save {key} to database: {e}")
                    # ===== END DATABASE SAVE =====
                    
                    # Still store in detected_jumps for immediate visualization
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

# ------------------------------------------------------------------
# Populate comparison athlete dropdown from database
# ------------------------------------------------------------------
@app.callback(
    Output("compare_athlete", "options"),
    Input("detect_button", "n_clicks"),
    Input("load_btn", "n_clicks"),  # ← ADD THIS as trigger
    prevent_initial_call=False  # ← Make sure it runs on startup
)
def populate_comparison_athletes(detect_clicks, load_clicks):
    """Load all athletes from database for comparison."""
    print("[DEBUG] Populate comparison athletes callback fired")
    try:
        athletes = get_all_athletes()
        print(f"[DEBUG] Loaded {len(athletes)} athletes: {athletes}")
        options = [{"label": a, "value": a} for a in athletes]
        print(f"[DEBUG] Returning {len(options)} options")
        return options
    except Exception as e:
        print(f"[ERROR] Failed to populate athletes: {e}")
        import traceback
        traceback.print_exc()
        return []
    
# ------------------------------------------------------------------
# Display focus athlete name in Vertical Vector tab
# ------------------------------------------------------------------
@app.callback(
    Output("focus_athlete_display", "children"),
    Input("athlete_selected", "value")
)
def show_focus_athlete(athletes):
    """Display which athlete is currently being analyzed."""
    if not athletes:
        return html.Div("⚠️ Please select an athlete in the main dropdown", 
                       style={"color": "orange", "fontSize": "16px"})
    
    if isinstance(athletes, list):
        athlete_name = athletes[0]  # Use first selected athlete as focus
    else:
        athlete_name = athletes
    
    return html.Div([
        html.Span("📊 Analyzing: ", style={"fontSize": "16px"}),
        html.Span(athlete_name, style={"fontSize": "18px", "fontWeight": "bold", "color": "#1f77b4"})
    ])

# ------------------------------------------------------------------
# Population comparison visualization - AUTO DISPLAY
# ------------------------------------------------------------------
@app.callback(
    Output("comparison_plot", "figure"),
    Output("population_stats_table", "data"),
    Input("athlete_selected", "value"),  # ← Auto-trigger on athlete selection
    Input("compare_btn", "n_clicks"),     # ← Also trigger on button click
    State("compare_athlete", "value"),
    State("compare_metric", "value"),
    prevent_initial_call=False  # ← Allow initial call
)
def show_population_comparison(focus_athletes, n_clicks, compare_athlete, metric):
    """Create population comparison visualization with focus athlete."""
    
    # Get focus athlete
    if not focus_athletes:
        return go.Figure(), []  # ← Return empty list, not None
    
    if isinstance(focus_athletes, list):
        focus_athlete = focus_athletes[0]
    else:
        focus_athlete = focus_athletes
    
    if not metric:
        metric = "jump_height_cm"  # ← Default metric
    
    # Load all jumps from database
    df_all = get_all_jumps_dataframe()
    
    if df_all.empty:
        return go.Figure(), []  # ← Return empty list
    
    # Get focus athlete's data
    df_focus = df_all[df_all['athlete_name'] == focus_athlete].copy()
    
    if df_focus.empty:
        return go.Figure(), []  # ← Return empty list
    
    # Get population data
    df_population = df_all[df_all['athlete_name'] != focus_athlete]
    
    # Convert dates and sort
    df_focus['date'] = pd.to_datetime(df_focus['session_date'], format='%Y.%m.%d', errors='coerce')
    df_focus = df_focus.sort_values('date')
    
    # Remove NaN values
    df_focus = df_focus.dropna(subset=[metric, 'date'])
    df_population = df_population.dropna(subset=[metric])
    
    if df_population.empty:
        # Still show focus athlete even without population
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_focus['date'],
            y=df_focus[metric],
            mode='lines+markers',
            name=focus_athlete,
            marker=dict(size=10, color='blue'),
            line=dict(width=3, color='blue')
        ))
        
        metric_labels = {
            "jump_height_cm": "Jump Height (cm)",
            "rsi_modified": "RSI-modified",
            "peak_power_rel": "Peak Power (W/kg)",
            "mean_power_rel": "Mean Power (W/kg)",
            "takeoff_velocity": "Takeoff Velocity (m/s)",
            "contraction_time": "Contraction Time (s)",
            "prop_max_force_pct": "Peak Propulsive Force (% BW)"
        }
        
        fig.update_layout(
            title=f"{focus_athlete} - {metric_labels.get(metric, metric)}",
            xaxis_title="Date",
            yaxis_title=metric_labels.get(metric, metric),
            template="plotly_white",
            showlegend=True,
            height=500
        )
        
        return fig, []  # ← Return empty list
    
    # Calculate percentiles
    p0 = df_population[metric].min()
    p25 = df_population[metric].quantile(0.25)
    p50 = df_population[metric].quantile(0.50)
    p75 = df_population[metric].quantile(0.75)
    p85 = df_population[metric].quantile(0.85)
    p95 = df_population[metric].quantile(0.95)
    p100 = df_population[metric].max()
    
    # Create figure
    fig = go.Figure()
    
    # Add percentile background bands
    fig.add_hrect(y0=p0, y1=p25, fillcolor="green", opacity=0.1, layer="below", line_width=0)
    fig.add_hrect(y0=p25, y1=p50, fillcolor="yellow", opacity=0.1, layer="below", line_width=0)
    fig.add_hrect(y0=p50, y1=p75, fillcolor="orange", opacity=0.1, layer="below", line_width=0)
    fig.add_hrect(y0=p75, y1=p85, fillcolor="darkorange", opacity=0.1, layer="below", line_width=0)
    fig.add_hrect(y0=p85, y1=p95, fillcolor="orangered", opacity=0.1, layer="below", line_width=0)
    fig.add_hrect(y0=p95, y1=p100, fillcolor="red", opacity=0.1, layer="below", line_width=0)
    
    # Add percentile reference lines
    fig.add_hline(y=p50, line_dash="dash", line_color="gray", line_width=1, 
                  annotation_text="50th", annotation_position="right")
    fig.add_hline(y=p75, line_dash="dot", line_color="orange", line_width=1,
                  annotation_text="75th", annotation_position="right")
    fig.add_hline(y=p95, line_dash="dot", line_color="red", line_width=1,
                  annotation_text="95th", annotation_position="right")
    
    # Add FOCUS athlete (BLUE - PRIMARY)
    fig.add_trace(go.Scatter(
        x=df_focus['date'],
        y=df_focus[metric],
        mode='lines+markers',
        name=f"🎯 {focus_athlete}",
        marker=dict(size=10, color='blue'),
        line=dict(width=3, color='blue')
    ))
    
    # Add COMPARISON athlete if selected (RED - SECONDARY)
    if compare_athlete and compare_athlete != focus_athlete:
        df_compare = df_all[df_all['athlete_name'] == compare_athlete].copy()
        
        if not df_compare.empty:
            df_compare['date'] = pd.to_datetime(df_compare['session_date'], format='%Y.%m.%d', errors='coerce')
            df_compare = df_compare.sort_values('date')
            df_compare = df_compare.dropna(subset=[metric, 'date'])
            
            if not df_compare.empty:
                fig.add_trace(go.Scatter(
                    x=df_compare['date'],
                    y=df_compare[metric],
                    mode='lines+markers',
                    name=f"⚡ {compare_athlete}",
                    marker=dict(size=8, color='red', symbol='diamond'),
                    line=dict(width=2, color='red', dash='dash')
                ))
    
    # Metric labels
    metric_labels = {
        "jump_height_cm": "Jump Height (cm)",
        "rsi_modified": "RSI-modified",
        "peak_power_rel": "Peak Power (W/kg)",
        "mean_power_rel": "Mean Power (W/kg)",
        "takeoff_velocity": "Takeoff Velocity (m/s)",
        "contraction_time": "Contraction Time (s)",
        "prop_max_force_pct": "Peak Propulsive Force (% BW)"
    }
    
    title = f"🎯 {focus_athlete} - {metric_labels.get(metric, metric)}"
    if compare_athlete and compare_athlete != focus_athlete:
        title += f" vs ⚡ {compare_athlete}"
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=metric_labels.get(metric, metric),
        template="plotly_white",
        showlegend=True,
        height=500
    )
    
    # Calculate statistics
    pop_mean = df_population[metric].mean()
    focus_mean = df_focus[metric].mean()
    focus_best = df_focus[metric].max()
    focus_latest = df_focus[metric].iloc[-1] if len(df_focus) > 0 else 0
    percentile = (df_population[metric] < focus_best).sum() / len(df_population) * 100
    
    # Build stats table
    stats_data = [{
        "Athlete": f"🎯 {focus_athlete}",
        "Metric": metric_labels.get(metric, metric),
        "Latest": f"{focus_latest:.2f}",
        "Best": f"{focus_best:.2f}",
        "Mean": f"{focus_mean:.2f}",
        "Pop. Median": f"{p50:.2f}",
        "Percentile": f"{percentile:.0f}%",
        "Jumps": len(df_focus)
    }]
    
    # Add comparison athlete stats
    if compare_athlete and compare_athlete != focus_athlete:
        df_compare_full = df_all[df_all['athlete_name'] == compare_athlete]
        if not df_compare_full.empty:
            df_compare_metric = df_compare_full.dropna(subset=[metric])
            if not df_compare_metric.empty:
                compare_mean = df_compare_metric[metric].mean()
                compare_best = df_compare_metric[metric].max()
                compare_latest = df_compare_metric[metric].iloc[-1]
                compare_percentile = (df_population[metric] < compare_best).sum() / len(df_population) * 100
                
                stats_data.append({
                    "Athlete": f"⚡ {compare_athlete}",
                    "Metric": metric_labels.get(metric, metric),
                    "Latest": f"{compare_latest:.2f}",
                    "Best": f"{compare_best:.2f}",
                    "Mean": f"{compare_mean:.2f}",
                    "Pop. Median": f"{p50:.2f}",
                    "Percentile": f"{compare_percentile:.0f}%",
                    "Jumps": len(df_compare_metric)
                })
    
    return fig, stats_data


# ------------------------------------------------------------------
# Pizza Plot - Show BOTH athletes with absolute values
# ------------------------------------------------------------------
@app.callback(
    Output("pizza_plot", "figure"),
    Input("athlete_selected", "value"),
    Input("compare_btn", "n_clicks"),
    State("compare_athlete", "value"),
    prevent_initial_call=False
)
def update_pizza_plot(focus_athletes, n_clicks, compare_athlete):
    """Create pizza plot showing focus athlete and optional comparison."""
    
    if not focus_athletes:
        return go.Figure()
    
    if isinstance(focus_athletes, list):
        focus_athlete = focus_athletes[0]
    else:
        focus_athlete = focus_athletes
    
    # Load all jumps
    df_all = get_all_jumps_dataframe()
    
    if df_all.empty:
        return go.Figure()
    
    # Get focus athlete data
    df_focus = df_all[df_all['athlete_name'] == focus_athlete]
    df_population = df_all[df_all['athlete_name'] != focus_athlete]
    
    if df_focus.empty or df_population.empty:
        return go.Figure()
    
    # Define metrics (column_name, display_label, lower_is_better)
    metrics_config = [
        ("jump_height_cm", "Jump Height (cm)", False),
        ("rsi_modified", "RSI-mod", False),
        ("peak_power_rel", "Peak Power (W/kg)", False),
        ("takeoff_velocity", "Takeoff Vel (m/s)", False),
        ("prop_max_force_pct", "Peak Force (% BW)", False),
        ("contraction_time", "Contract Time (s)", True),  # ← Lower is better
        ("brk_time", "Braking Time (s)", True),           # ← Lower is better
        ("prop_time", "Propulsive Time (s)", True),       # ← Lower is better
        ("cmj_depth_cm", "CMJ Depth (cm)", False)
    ]
    
    fig = go.Figure()
    
    # ===== FOCUS ATHLETE (BLUE) =====
    percentiles_focus = []
    labels = []
    hover_texts_focus = []
    
    for metric_col, label, lower_is_better in metrics_config:
        if metric_col not in df_focus.columns or metric_col not in df_population.columns:
            continue
        
        athlete_values = df_focus[metric_col].dropna()
        pop_values = df_population[metric_col].dropna()
        
        if len(athlete_values) == 0 or len(pop_values) == 0:
            continue
        
        athlete_best = athlete_values.max() if not lower_is_better else athlete_values.min()
        
        # Calculate percentile (INVERTED for time metrics)
        if lower_is_better:
            percentile = (pop_values > athlete_best).sum() / len(pop_values) * 100
        else:
            percentile = (pop_values < athlete_best).sum() / len(pop_values) * 100
        
        percentiles_focus.append(percentile)
        labels.append(label)
        hover_texts_focus.append(f"{label}<br>Value: {athlete_best:.2f}<br>Percentile: {percentile:.0f}%")
    
    if len(percentiles_focus) > 0:
        percentiles_closed = percentiles_focus + [percentiles_focus[0]]
        labels_closed = labels + [labels[0]]
        hover_closed_focus = hover_texts_focus + [hover_texts_focus[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=percentiles_closed,
            theta=labels_closed,
            fill='toself',
            name=f"🎯 {focus_athlete}",
            fillcolor='rgba(0, 100, 255, 0.3)',
            line=dict(color='blue', width=3),
            hovertext=hover_closed_focus,
            hoverinfo='text'
        ))
    
    # ===== COMPARISON ATHLETE (RED) =====
    if compare_athlete and compare_athlete != focus_athlete:
        df_compare = df_all[df_all['athlete_name'] == compare_athlete]
        
        if not df_compare.empty:
            percentiles_compare = []
            hover_texts_compare = []
            
            for metric_col, label, lower_is_better in metrics_config:
                if metric_col not in df_compare.columns:
                    continue
                
                compare_values = df_compare[metric_col].dropna()
                pop_values = df_population[metric_col].dropna()
                
                if len(compare_values) == 0 or len(pop_values) == 0:
                    continue
                
                compare_best = compare_values.max() if not lower_is_better else compare_values.min()
                
                if lower_is_better:
                    percentile = (pop_values > compare_best).sum() / len(pop_values) * 100
                else:
                    percentile = (pop_values < compare_best).sum() / len(pop_values) * 100
                
                percentiles_compare.append(percentile)
                hover_texts_compare.append(f"{label}<br>Value: {compare_best:.2f}<br>Percentile: {percentile:.0f}%")
            
            if len(percentiles_compare) == len(percentiles_focus):
                percentiles_compare_closed = percentiles_compare + [percentiles_compare[0]]
                hover_closed_compare = hover_texts_compare + [hover_texts_compare[0]]
                
                fig.add_trace(go.Scatterpolar(
                    r=percentiles_compare_closed,
                    theta=labels_closed,
                    fill='toself',
                    name=f"⚡ {compare_athlete}",
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    line=dict(color='red', width=2, dash='dash'),
                    hovertext=hover_closed_compare,
                    hoverinfo='text'
                ))
    
    title = f"🎯 {focus_athlete} - Percentile Profile"
    if compare_athlete and compare_athlete != focus_athlete:
        title = f"🎯 {focus_athlete} vs ⚡ {compare_athlete}"
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[0, 25, 50, 75, 100],
                ticktext=['0%', '25%', '50%', '75%', '100%'],
                gridcolor='lightgray'
            ),
            angularaxis=dict(
                gridcolor='lightgray'
            )
        ),
        showlegend=True,
        title=title,
        height=500
    )
    
    return fig

if __name__ == "__main__":
    app.run(debug=True)