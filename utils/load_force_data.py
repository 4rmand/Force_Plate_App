import pandas as pd
import re
import os
from typing import Tuple, Dict, List

def load_force_data(filepath: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load and clean a single ForceDecks CSV file."""
    metadata = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    header_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Time"):
            header_index = i
            break

    if header_index is None:
        raise ValueError(f"No 'Time' header found in {filepath}")

    # --- Parse metadata lines
    for line in lines[:header_index]:
        parts = [p.strip() for p in line.strip().split(',') if p.strip() != ""]
        if len(parts) >= 2:
            key = parts[0]
            value = ",".join(parts[1:]).replace('"', '').replace("'", "")
            metadata[key] = value

    # --- Load data section
    df = pd.read_csv(filepath, skiprows=header_index, delimiter=',', skip_blank_lines=True)
    df.columns = [re.sub(r'\s+', ' ', c.strip()) for c in df.columns]

    # Convert decimal commas to dots
    for col in df.columns:
        df[col] = (
            df[col].astype(str)
            .str.replace(',', '.', regex=False)
            .str.replace('"', '', regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'Z Left' in df.columns and 'Z Right' in df.columns:
        df['Z Total'] = df['Z Left'] + df['Z Right']

    df = df.reset_index(drop=True)
    return df, metadata


def load_all_files_for_athlete(folder_path: str, athlete_name: str) -> Dict[str, Tuple[pd.DataFrame, Dict[str, str]]]:
    """
    Load all ForceDecks CSVs for one athlete based on their name.
    Returns a dictionary {filename: (df, metadata)}.
    """
    files = [f for f in os.listdir(folder_path) if athlete_name.lower() in f.lower() and f.endswith('.csv')]

    if not files:
        raise FileNotFoundError(f"No CSV files found for {athlete_name} in {folder_path}")

    all_data = {}
    for f in files:
        full_path = os.path.join(folder_path, f)
        df, meta = load_force_data(full_path)
        all_data[f] = (df, meta)

    return all_data

if __name__ == "__main__":
    folder = "/Users/Armand/Desktop/Python/VALD"
    athlete = "Nolan TRAORE"
    data = load_all_files_for_athlete(folder, athlete)

    for filename, (df, meta) in data.items():
        print(f"✅ Loaded {filename} → {df.shape} rows")
