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
    
    Args:
        jump_id: Unique identifier (e.g., "athlete_date_file_jump1")
        athlete_name: Athlete name
        session_date: Date string (e.g., "2024.01.15")
        file_name: Original CSV filename
        jump_number: Jump number in that file
        metrics_dict: Dict with keys 'unw', 'brk', 'prop', 'overall'
        normalized_curves: Optional dict with 'force' and 'velocity' arrays
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Flatten metrics dict
    unw = metrics_dict.get('unw', {})
    brk = metrics_dict.get('brk', {})
    prop = metrics_dict.get('prop', {})
    overall = metrics_dict.get('overall', {})
    
    # Extract overall metrics
    bw = overall.get('Body Weight (N)', metrics_dict.get('body_weight', 0))
    
    # Insert or replace jump metrics
    cursor.execute("""
        INSERT OR REPLACE INTO jump_metrics VALUES (
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            CURRENT_TIMESTAMP
        )
    """, (
        jump_id, athlete_name, session_date, file_name, jump_number,
        # Overall
        bw,
        overall.get('Flight Time (s)', 0),
        overall.get('Jump Height (m)', 0),
        overall.get('Jump Height (cm)', 0),
        overall.get('Takeoff Velocity (m/s)', 0),
        overall.get('Contraction Time (s)', 0),
        overall.get('RSI-modified', 0),
        overall.get('Mean Power (W)', 0),
        overall.get('Peak Power (W)', 0),
        overall.get('Mean Power (W/kg)', 0),
        overall.get('Peak Power (W/kg)', 0),
        # Unweighting
        unw.get('Time (s)', 0),
        unw.get('Min Force (N)', 0),
        unw.get('Min Force (% BW)', 0),
        unw.get('Impulse (N·s)', 0),
        unw.get('Min Force Left (N)') if unw.get('Min Force Left (N)') != "N/A" else None,
        unw.get('Min Force Right (N)') if unw.get('Min Force Right (N)') != "N/A" else None,
        unw.get('Min Force Asym (%)') if unw.get('Min Force Asym (%)') != "N/A" else None,
        # Braking
        brk.get('Time (s)', 0),
        brk.get('Max Force (N)', 0),
        brk.get('Max Force (% BW)', 0),
        brk.get('Mean Force (N)', 0),
        brk.get('Min Velocity (m/s)', 0),
        brk.get('Impulse (N·s)', 0),
        brk.get('Mean Power (W)', 0),
        brk.get('Peak Power (W)', 0),
        brk.get('Max Force Left (N)') if brk.get('Max Force Left (N)') != "N/A" else None,
        brk.get('Max Force Right (N)') if brk.get('Max Force Right (N)') != "N/A" else None,
        brk.get('Max Force Asym (%)') if brk.get('Max Force Asym (%)') != "N/A" else None,
        # Propulsive
        prop.get('Time (s)', 0),
        prop.get('Max Force (N)', 0),
        prop.get('Max Force (% BW)', 0),
        prop.get('Mean Force (N)', 0),
        prop.get('Max Velocity (m/s)', 0),
        prop.get('Takeoff Velocity (m/s)', 0),
        prop.get('Impulse (N·s)', 0),
        prop.get('Mean Power (W)', 0),
        prop.get('Peak Power (W)', 0),
        prop.get('Max Force Left (N)') if prop.get('Max Force Left (N)') != "N/A" else None,
        prop.get('Max Force Right (N)') if prop.get('Max Force Right (N)') != "N/A" else None,
        prop.get('Max Force Asym (%)') if prop.get('Max Force Asym (%)') != "N/A" else None,
        prop.get('Mean Power Left (W)') if prop.get('Mean Power Left (W)') != "N/A" else None,
        prop.get('Mean Power Right (W)') if prop.get('Mean Power Right (W)') != "N/A" else None,
        prop.get('Mean Power Asym (%)') if prop.get('Mean Power Asym (%)') != "N/A" else None,
        prop.get('Peak Power Left (W)') if prop.get('Peak Power Left (W)') != "N/A" else None,
        prop.get('Peak Power Right (W)') if prop.get('Peak Power Right (W)') != "N/A" else None,
        prop.get('Peak Power Asym (%)') if prop.get('Peak Power Asym (%)') != "N/A" else None,
    ))
    
    # Optionally save normalized curves
    if normalized_curves:
        force_json = json.dumps(normalized_curves.get('force', []))
        vel_json = json.dumps(normalized_curves.get('velocity', []))
        
        cursor.execute("""
            INSERT OR REPLACE INTO jump_curves VALUES (?, ?, ?)
        """, (jump_id, force_json, vel_json))
    
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
    return dfs