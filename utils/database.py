import sqlite3
import json
import pandas as pd
from pathlib import Path

DB_PATH = "cmj_data.db"

def init_database():
    """Initialize the SQLite database with required tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Table 1: Jump Metrics (one row per jump)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS jump_metrics (
            jump_id TEXT PRIMARY KEY,
            athlete_name TEXT NOT NULL,
            session_date TEXT NOT NULL,
            file_name TEXT NOT NULL,
            jump_number INTEGER NOT NULL,
            
            -- Overall metrics
            body_weight REAL,
            flight_time REAL,
            jump_height_m REAL,
            jump_height_cm REAL,
            takeoff_velocity REAL,
            contraction_time REAL,
            rsi_modified REAL,
            mean_power_total REAL,
            peak_power_total REAL,
            mean_power_rel REAL,
            peak_power_rel REAL,
            cmj_depth_m REAL,       
            cmj_depth_cm REAL,       
            
            -- Unweighting phase
            unw_time REAL,
            unw_min_force REAL,
            unw_min_force_pct REAL,
            unw_impulse REAL,
            unw_min_force_left REAL,
            unw_min_force_right REAL,
            unw_min_force_asym REAL,
            
            -- Braking phase
            brk_time REAL,
            brk_max_force REAL,
            brk_max_force_pct REAL,
            brk_mean_force REAL,
            brk_min_velocity REAL,
            brk_impulse REAL,
            brk_mean_power REAL,
            brk_peak_power REAL,
            brk_max_force_left REAL,
            brk_max_force_right REAL,
            brk_max_force_asym REAL,
            
            -- Propulsive phase
            prop_time REAL,
            prop_max_force REAL,
            prop_max_force_pct REAL,
            prop_mean_force REAL,
            prop_max_velocity REAL,
            prop_takeoff_velocity REAL,
            prop_impulse REAL,
            prop_mean_power REAL,
            prop_peak_power REAL,
            prop_max_force_left REAL,
            prop_max_force_right REAL,
            prop_max_force_asym REAL,
            prop_mean_power_left REAL,
            prop_mean_power_right REAL,
            prop_mean_power_asym REAL,
            prop_peak_power_left REAL,
            prop_peak_power_right REAL,
            prop_peak_power_asym REAL,
            
            -- Metadata
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            UNIQUE(athlete_name, session_date, file_name, jump_number)
        )
    """)
    
    # Table 2: Normalized Force Curves (optional - for visualization)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS jump_curves (
            jump_id TEXT PRIMARY KEY,
            force_curve_json TEXT,  -- Normalized 0-100% time, force values
            velocity_curve_json TEXT,  -- Normalized velocity
            FOREIGN KEY (jump_id) REFERENCES jump_metrics(jump_id)
        )
    """)
    
    conn.commit()
    conn.close()
    print("✅ Database initialized")


def save_jump_to_db(jump_id, athlete_name, session_date, file_name, jump_number, 
                    metrics_dict, normalized_curves=None):
    """
    Save a single jump's metrics (and optionally curves) to database.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Flatten metrics dict
    unw = metrics_dict.get('unw', {})
    brk = metrics_dict.get('brk', {})
    prop = metrics_dict.get('prop', {})
    overall = metrics_dict.get('overall', {})
    
    # Extract values with safe defaults
    def safe_get(d, key, default=0):
        val = d.get(key, default)
        return None if val == "N/A" else val
    
    # Build the values tuple
    values = (
        # Basic info (5)
        jump_id, 
        athlete_name, 
        session_date, 
        file_name, 
        jump_number,
        
        # Overall metrics (13)
        metrics_dict.get('body_weight', 0),
        metrics_dict.get('flight_time', 0),
        safe_get(overall, 'Jump Height (m)'),
        safe_get(overall, 'Jump Height (cm)'),
        safe_get(overall, 'Takeoff Velocity (m/s)'),
        safe_get(overall, 'Contraction Time (s)'),
        safe_get(overall, 'RSI-modified'),
        safe_get(overall, 'Mean Power (W)'),
        safe_get(overall, 'Peak Power (W)'),
        safe_get(overall, 'Mean Power (W/kg)'),
        safe_get(overall, 'Peak Power (W/kg)'),
        safe_get(overall, 'CMJ Depth (m)'),
        safe_get(overall, 'CMJ Depth (cm)'),
        
        # Unweighting (7)
        safe_get(unw, 'Time (s)'),
        safe_get(unw, 'Min Force (N)'),
        safe_get(unw, 'Min Force (% BW)'),
        safe_get(unw, 'Impulse (N·s)'),
        safe_get(unw, 'Min Force Left (N)'),
        safe_get(unw, 'Min Force Right (N)'),
        safe_get(unw, 'Min Force Asym (%)'),
        
        # Braking (11)
        safe_get(brk, 'Time (s)'),
        safe_get(brk, 'Max Force (N)'),
        safe_get(brk, 'Max Force (% BW)'),
        safe_get(brk, 'Mean Force (N)'),
        safe_get(brk, 'Min Velocity (m/s)'),
        safe_get(brk, 'Impulse (N·s)'),
        safe_get(brk, 'Mean Power (W)'),
        safe_get(brk, 'Peak Power (W)'),
        safe_get(brk, 'Max Force Left (N)'),
        safe_get(brk, 'Max Force Right (N)'),
        safe_get(brk, 'Max Force Asym (%)'),
        
        # Propulsive (18)
        safe_get(prop, 'Time (s)'),
        safe_get(prop, 'Max Force (N)'),
        safe_get(prop, 'Max Force (% BW)'),
        safe_get(prop, 'Mean Force (N)'),
        safe_get(prop, 'Max Velocity (m/s)'),
        safe_get(prop, 'Takeoff Velocity (m/s)'),
        safe_get(prop, 'Impulse (N·s)'),
        safe_get(prop, 'Mean Power (W)'),
        safe_get(prop, 'Peak Power (W)'),
        safe_get(prop, 'Max Force Left (N)'),
        safe_get(prop, 'Max Force Right (N)'),
        safe_get(prop, 'Max Force Asym (%)'),
        safe_get(prop, 'Mean Power Left (W)'),
        safe_get(prop, 'Mean Power Right (W)'),
        safe_get(prop, 'Mean Power Asym (%)'),
        safe_get(prop, 'Peak Power Left (W)'),
        safe_get(prop, 'Peak Power Right (W)'),
        safe_get(prop, 'Peak Power Asym (%)'),
    )
    
    # DEBUG: Print counts
    print(f"[DEBUG] Number of values in tuple: {len(values)}")
    print(f"[DEBUG] Basic: 5, Overall: 13, Unw: 7, Brk: 11, Prop: 18 = {5+13+7+11+18}")
    
    # Insert or replace jump metrics
    cursor.execute("""
        INSERT OR REPLACE INTO jump_metrics VALUES (
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            CURRENT_TIMESTAMP
        )
    """, values)
    
    conn.commit()
    conn.close()


def get_all_athletes():
    """Get list of all athletes in database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT athlete_name FROM jump_metrics ORDER BY athlete_name")
    athletes = [row[0] for row in cursor.fetchall()]
    conn.close()
    return athletes


def get_athlete_jumps(athlete_name):
    """Get all jumps for a specific athlete."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM jump_metrics WHERE athlete_name = ? ORDER BY session_date DESC",
        conn,
        params=(athlete_name,)
    )
    conn.close()
    return df


def get_population_stats(metric_column):
    """
    Get population statistics for a specific metric.
    Returns mean, std, percentiles.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(f"""
        SELECT 
            AVG({metric_column}) as mean,
            MAX({metric_column}) as max,
            MIN({metric_column}) as min
        FROM jump_metrics
        WHERE {metric_column} IS NOT NULL
    """)
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            'mean': result[0],
            'max': result[1],
            'min': result[2]
        }
    return None


def get_all_jumps_dataframe():
    """Load all jumps as a pandas DataFrame for analysis."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM jump_metrics ORDER BY athlete_name, session_date", conn)
    conn.close()
    return df