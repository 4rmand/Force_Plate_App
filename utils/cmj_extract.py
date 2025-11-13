import numpy as np
import pandas as pd
from scipy import signal
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

_MIN_MASS_KG = 40
_MIN_BW_STABLE_S = 1.0

def _find_stable_bw(z: np.ndarray, fs: int, min_len_s: float = 1.0) -> float:
    w = int(0.05 * fs) // 2 * 2 + 1
    z_smooth = pd.Series(z).rolling(w, center=True).mean().to_numpy()
    std_w = int(0.2 * fs)
    rolling_std = pd.Series(z_smooth).rolling(std_w, center=True).std().to_numpy()
    stable = rolling_std < 5.0
    min_len = int(min_len_s * fs)
    for start in range(len(stable) - min_len):
        if np.all(stable[start:start + min_len]):
            return float(np.median(z_smooth[start:start + min_len]))
    return float(np.median(z_smooth[:fs]))


def extract_cmj(df: pd.DataFrame, fs: int = None, debug: bool = True,
                min_drop_N: float = 100.0,
                min_recovery_duration_s: float = 0.8,
                max_flight_s: float = 2.0):
    """
    Detect CMJs in a force-time file.
    """
    if fs is None:
        fs = int(1 / np.diff(df["Time"]).mean())

    # Extract force data
    z_left = df["Z Left"].astype(float).to_numpy() if "Z Left" in df.columns else np.zeros(len(df))
    z_right = df["Z Right"].astype(float).to_numpy() if "Z Right" in df.columns else np.zeros(len(df))
    z_total = z_left + z_right
    time = df["Time"].to_numpy()

    # Check if athlete is on plate
    min_force = _MIN_MASS_KG * 9.81
    on_plate = np.where(z_total >= min_force)[0]
    if on_plate.size == 0:
        if debug:
            print("[CMJ] No one detected on plate (force < threshold).")
        return []
    first_on = on_plate[0]

    # Slice all arrays from first_on onwards
    z_total_full = z_total.copy()
    z_total = z_total[first_on:]
    z_left = z_left[first_on:]
    z_right = z_right[first_on:]
    time = time[first_on:]

    # Estimate body weight
    bw = _find_stable_bw(z_total, fs, min_len_s=_MIN_BW_STABLE_S)
    if debug:
        print(f"[CMJ] Estimated BW = {bw:.1f} N")

    # Smooth force signal
    sos = signal.butter(2, 20, btype="low", fs=fs, output="sos")
    zf = signal.sosfiltfilt(sos, z_total)

    # Detect flight phases
    flight = zf < 20
    d = np.diff(flight.astype(int))
    to = np.where(d == 1)[0]
    ld = np.where(d == -1)[0]
    
    if ld.size and to.size and ld[0] < to[0]:
        ld = ld[1:]
    
    n_pairs = min(len(to), len(ld))
    pairs = np.column_stack([to[:n_pairs], ld[:n_pairs]])
    
    min_flight_pts = int(0.15 * fs)
    pairs = pairs[(pairs[:, 1] - pairs[:, 0]) >= min_flight_pts]

    jumps = []

    for j_idx, (to_idx, ld_idx) in enumerate(pairs):
        flight_time = (ld_idx - to_idx) / fs
        
        # ===== CHECK 1: Unrealistic flight time =====
        if flight_time > max_flight_s:
            if debug:
                print(f"[CMJ {j_idx}] ❌ Unrealistic flight ({flight_time:.2f}s) — probably off plate")
            continue

# ===== CHECK 2: Prevent rebound jumps (dual approach) =====
        
        # Part A: Check time since last ACCEPTED jump's landing
        min_time_after_landing = 2.0
        
        if len(jumps) > 0:
            last_accepted_landing = jumps[-1][1]['landing_idx_global'] - first_on
            time_since_last_landing = (to_idx - last_accepted_landing) / fs
            
            if time_since_last_landing < min_time_after_landing:
                if debug:
                    print(f"[CMJ {j_idx}] ❌ Rebound jump - only {time_since_last_landing:.2f}s since last landing")
                continue
        
        # Part B: Check for immediate rebound after THIS jump's landing
        # (force drops >20N within 0.3s after landing = another countermovement starting)
        rebound_check_window = int(0.3 * fs)
        rebound_check_start = ld_idx + int(0.1 * fs)
        rebound_check_end = min(len(zf), rebound_check_start + rebound_check_window)
        
        if rebound_check_end > rebound_check_start:
            force_after_landing = zf[rebound_check_start:rebound_check_end]
            
            # Find stable force around BW
            stable_force = force_after_landing[(force_after_landing > 0.5 * bw) & 
                                                (force_after_landing < 8 * bw)]
            
            if len(stable_force) > 0:
                baseline_force = np.median(stable_force)
                min_force_in_window = np.min(force_after_landing)
                force_drop = baseline_force - min_force_in_window
                
                if min_force_in_window < 20:
                    if debug:
                        print(f"[CMJ {j_idx}] ❌ Immediate rebound - force drops to {min_force_in_window:.0f}N within 0.3s after landing")
                    continue

        # ===== CHECK 3: Drop jump detection =====
        pre_win_start = max(0, int(to_idx - 0.4 * fs))
        pre_win_end = int(to_idx - 0.1 * fs)
        if pre_win_end <= pre_win_start:
            mean_pre = np.mean(zf[pre_win_start:to_idx]) if to_idx > pre_win_start else np.mean(zf[:to_idx])
        else:
            mean_pre = np.mean(zf[pre_win_start:pre_win_end])

        if mean_pre < 0.5 * bw:
            if debug:
                print(f"[CMJ {j_idx}] ❌ Drop Jump — pre-force {mean_pre:.1f}N < 50% BW")
            continue

        # ===== CHECK 4: Find unweighting start =====
        unweight_mask = zf < (bw - 20)
        start_pos = None
        min_consec = max(5, int(0.01 * fs))
        search_start = max(0, to_idx - int(1.0 * fs))

        for i in range(search_start, to_idx - min_consec):
            if np.all(unweight_mask[i:i + min_consec]):
                check_end = min(len(zf), i + int(0.2 * fs))
                min_after = zf[i:check_end].min() if check_end > i else zf[i]
                if (zf[i] - min_after) >= min_drop_N:
                    start_pos = i
                    break

        if start_pos is None:
            if debug:
                print(f"[CMJ {j_idx}] ❌ Unweight start not robust (no sustained drop)")
            continue

        # ===== NEW CHECK: Validate sufficient countermovement (reject Squat Jumps) =====
        # Check that force drops at least 300N from start_pos onwards
        force_at_start = zf[start_pos]
        min_force_after_start = np.min(zf[start_pos:to_idx])
        countermovement_depth = force_at_start - min_force_after_start

        if countermovement_depth < 300:  # Require at least 300N drop for CMJ
            if debug:
                print(f"[CMJ {j_idx}] ❌ Insufficient countermovement ({countermovement_depth:.0f}N drop) — likely Squat Jump")
            continue

        # Also check velocity: must reach at least -0.5 m/s during eccentric phase
        # Compute velocity for this segment
        temp_df = pd.DataFrame({
            'Time': time[start_pos:to_idx],
            'Z Total': zf[start_pos:to_idx]
        })
        temp_df = temp_df.reset_index(drop=True)
        temp_df_vel = compute_acc_vel(temp_df, fs, bw)
        min_velocity = np.min(temp_df_vel['Vel'].to_numpy())

        if min_velocity > -0.5:  # Must reach at least -0.5 m/s downward
            if debug:
                print(f"[CMJ {j_idx}] ❌ Insufficient eccentric velocity ({min_velocity:.2f} m/s) — likely Squat Jump")
            continue

        if debug:
            print(f"[CMJ {j_idx}] ✓ Valid countermovement: {countermovement_depth:.0f}N drop, peak ecc vel = {min_velocity:.2f} m/s")

        # ===== CHECK 4: Require 1s stable standing BEFORE unweighting =====
        stable_window_s = 1.0
        stable_window_pts = int(stable_window_s * fs)
        stable_thresh_low = 0.9 * bw
        stable_thresh_high = 1.1 * bw

        if start_pos < stable_window_pts:
            if debug:
                print(f"[CMJ {j_idx}] ❌ Not enough data before unweight start (need 1s stable)")
            continue

        pre_unweight_seg = zf[start_pos - stable_window_pts:start_pos]
        is_stable = np.all((pre_unweight_seg > stable_thresh_low) & (pre_unweight_seg < stable_thresh_high))

        if not is_stable:
            if debug:
                mean_pre_unweight = np.mean(pre_unweight_seg)
                min_pre_unweight = np.min(pre_unweight_seg)
                print(f"[CMJ {j_idx}] ❌ No stable 1s standing before unweight (mean={mean_pre_unweight:.0f}N [{mean_pre_unweight/bw*100:.0f}%], min={min_pre_unweight:.0f}N)")
            continue

        # ===== CHECK 5: Single-leg jump detection =====
        if "Z Left" in df.columns and "Z Right" in df.columns:
            left_seg = z_left[start_pos:to_idx]
            right_seg = z_right[start_pos:to_idx]
            
            if len(left_seg) == 0 or len(right_seg) == 0:
                if debug:
                    print(f"[CMJ {j_idx}] ❌ Empty leg segment during unweighting")
                continue
            
            mean_left = np.mean(left_seg)
            mean_right = np.mean(right_seg)
            
            if mean_left < 0.15 * bw or mean_right < 0.15 * bw:
                if debug:
                    pctL = (mean_left / bw) * 100
                    pctR = (mean_right / bw) * 100
                    print(f"[CMJ {j_idx}] ❌ Single-leg jump during unweighting (L={mean_left:.0f}N [{pctL:.0f}%], R={mean_right:.0f}N [{pctR:.0f}%])")
                continue

        # ===== CHECK 6: Post-landing recovery =====
        landing_end = ld_idx + int(0.1 * fs)
        recover_thresh = 0.8 * bw
        recovered = False
        recovery_start_idx = None
        
        for k in range(landing_end, len(zf) - int(min_recovery_duration_s * fs)):
            seg = zf[k:k + int(min_recovery_duration_s * fs)]
            if seg.size == 0:
                break
            if np.all(seg > recover_thresh):
                recovered = True
                recovery_start_idx = k
                break
        
        if not recovered:
            if debug:
                print(f"[CMJ {j_idx}] ❌ No stable recovery >80% BW for {min_recovery_duration_s}s after landing")
            continue

        # ===== ACCEPT THE JUMP =====
        start = max(0, int(start_pos - 1.0 * fs))
        end = min(len(zf), int(ld_idx + 0.5 * fs))
        sub = df.iloc[first_on + start:first_on + end].copy()
        sub = sub.reset_index(drop=True)
        sub["Time"] -= sub["Time"].iloc[0]

        rel_start = start
        rel_end = end
        sub_zf = zf[rel_start:rel_end]
        sub["Z Total"] = sub_zf
        sub["ForcePctBW"] = (sub["Z Total"] / bw) * 100
        
        if "Z Left" in df.columns and "Z Right" in df.columns:
            orig_sub = df.iloc[first_on + start:first_on + end].reset_index(drop=True)
            sub["Z Left"] = orig_sub.get("Z Left")
            sub["Z Right"] = orig_sub.get("Z Right")

        meta = dict(
            body_weight=float(bw),
            flight_time=float((ld_idx - to_idx) / fs),
            peak_force_pct=float((np.max(sub["Z Total"]) / bw) * 100),
            start_time=float(sub["Time"].iloc[0]),
            end_time=float(sub["Time"].iloc[-1]),
            start_idx_global=int(first_on + start),
            takeoff_idx_global=int(first_on + to_idx),
            landing_idx_global=int(first_on + ld_idx),
        )

        jumps.append((sub, meta))

        if debug:
            recovery_time = (recovery_start_idx - ld_idx) / fs if recovery_start_idx else 0
            print(f"[CMJ {j_idx}] ✅ ACCEPTED — flight={flight_time:.3f}s, 1s stable before unweight ✓, recovery after {recovery_time:.2f}s")

    if debug:
        print(f"[CMJ] ✅ Detected {len(jumps)} valid bilateral CMJ(s)")
    
    return jumps


def compute_acc_vel(df: pd.DataFrame, fs: int, bw: float, smooth_vel_s: float = 0.01):
    """Compute acceleration and velocity."""
    g = 9.81
    mass = bw / g
    df = df.copy()
    
    if "Z Total" not in df.columns:
        raise ValueError("compute_acc_vel: df must contain 'Z Total' column")
    
    df["Acc"] = (df["Z Total"] - bw) / mass
    df["Vel"] = cumulative_trapezoid(df["Acc"].to_numpy(), dx=1/fs, initial=0)

    win = max(1, int(smooth_vel_s * fs))
    if win > 1:
        df["Vel"] = pd.Series(df["Vel"]).rolling(win, center=True, min_periods=1).mean().to_numpy()

    return df


def detect_cmj_phases(df: pd.DataFrame, fs: int, bw: float, debug: bool = False):
    """
    Detect CMJ phases with robust detection against force plate fluctuations.
    """
    if "Vel" not in df.columns:
        if debug:
            print("[DEBUG] No Vel column in dataframe")
        return {}
    
    force = df["Z Total"].to_numpy()
    vel = df["Vel"].to_numpy()
    t = df["Time"].to_numpy()

    # 1. Unweighting start: force drops below 95% BW AND continues to drop at least 200N
    unweight_start_idx = None
    min_drop_required = 200  # Minimum drop to confirm it's real unweighting, not fluctuation
    
    for i in range(len(force) - int(0.3 * fs)):  # Need at least 0.3s of data ahead
        if force[i] < 0.95 * bw:
            # Check if force continues to drop by at least 200N in the next 0.3s
            future_window = force[i:i + int(0.3 * fs)]
            min_future_force = np.min(future_window)
            drop_magnitude = force[i] - min_future_force
            
            if drop_magnitude >= min_drop_required:
                unweight_start_idx = i
                if debug:
                    print(f"[DEBUG] Unweight start validated: force drops {drop_magnitude:.0f}N")
                break
    
    if unweight_start_idx is None:
        if debug:
            print(f"[DEBUG] No valid unweight start found (need force < 95% BW with 200N+ drop)")
            print(f"[DEBUG] Force range: {force.min():.0f}N to {force.max():.0f}N, BW={bw:.0f}N")
        return {}

    # 2. Find takeoff (force < 20N) - but must stay low for at least 50ms to avoid spikes
    takeoff_candidates = np.where(force < 20)[0]
    takeoff_idx = None
    min_flight_duration = int(0.05 * fs)  # 50ms minimum
    
    for candidate in takeoff_candidates:
        if candidate + min_flight_duration < len(force):
            # Check if force stays below 20N for at least 50ms
            if np.all(force[candidate:candidate + min_flight_duration] < 20):
                takeoff_idx = candidate
                break
    
    if takeoff_idx is None:
        if debug:
            print(f"[DEBUG] No takeoff found (force never < 20N for 50ms+)")
        return {}

    # 3. Find velocity zero-crossing BETWEEN unweight start and takeoff
    # Require that velocity was significantly negative before crossing (real eccentric phase)
    zero_cross_idx = None
    min_eccentric_velocity = -0.3  # Must reach at least -0.3 m/s to be valid eccentric phase
    
    for i in range(unweight_start_idx, takeoff_idx - 1):
        if vel[i] <= 0 and vel[i + 1] > 0:
            # Check if we had significant negative velocity before this crossing
            max_negative_vel = np.min(vel[unweight_start_idx:i+1])
            if max_negative_vel < min_eccentric_velocity:
                zero_cross_idx = i
                if debug:
                    print(f"[DEBUG] Zero crossing validated: peak eccentric vel = {max_negative_vel:.2f} m/s")
                break
    
    if zero_cross_idx is None:
        if debug:
            print(f"[DEBUG] No valid velocity zero crossing (need eccentric vel < -0.3 m/s)")
        return {}
    
    propulsive_start_idx = zero_cross_idx

    # 4. Find MINIMUM FORCE between unweight start and velocity zero-crossing
    # Use a smoothed version to avoid noise spikes
    force_segment = force[unweight_start_idx:propulsive_start_idx+1]
    if len(force_segment) < 5:
        if debug:
            print(f"[DEBUG] Force segment too short for braking detection")
        return {}
    
    # Smooth the segment slightly to avoid picking noise as minimum
    from scipy.ndimage import uniform_filter1d
    force_segment_smooth = uniform_filter1d(force_segment, size=min(5, len(force_segment)))
    braking_start_idx = unweight_start_idx + np.argmin(force_segment_smooth)

    # 5. Landing: force returns above 80% BW after takeoff (sustained for 50ms)
    landing_idx = None
    landing_threshold = 0.8 * bw
    min_landing_duration = int(0.05 * fs)
    
    for i in range(takeoff_idx, len(force) - min_landing_duration):
        if np.all(force[i:i + min_landing_duration] > landing_threshold):
            landing_idx = i
            break
    
    if landing_idx is None:
        if debug:
            print(f"[DEBUG] No landing found (force never > 80% BW for 50ms+)")
        return {}

    # 6. Peak force during propulsive phase (use smoothed to avoid noise spikes)
    force_propulsive = force[propulsive_start_idx:takeoff_idx]
    if len(force_propulsive) > 5:
        force_propulsive_smooth = uniform_filter1d(force_propulsive, size=5)
        peak_force_idx = propulsive_start_idx + np.argmax(force_propulsive_smooth)
    else:
        peak_force_idx = propulsive_start_idx + np.argmax(force_propulsive)

    phases_dict = dict(
        unweight_start=t[unweight_start_idx],
        braking_start=t[braking_start_idx],
        propulsive_start=t[propulsive_start_idx],
        peak_force=t[peak_force_idx],
        takeoff=t[takeoff_idx],
        landing=t[landing_idx],
    )
    
    if debug:
        print(f"[DEBUG] ✅ Phases detected successfully!")
        print(f"[DEBUG]   Unweight start: {t[unweight_start_idx]:.3f}s (force < 95% BW, drops 200N+)")
        print(f"[DEBUG]   Braking start: {t[braking_start_idx]:.3f}s (min force = {force[braking_start_idx]:.0f}N)")
        print(f"[DEBUG]   Propulsive start: {t[propulsive_start_idx]:.3f}s (vel crosses zero)")
        print(f"[DEBUG]   Peak force: {t[peak_force_idx]:.3f}s ({force[peak_force_idx]:.0f}N)")
        print(f"[DEBUG]   Takeoff: {t[takeoff_idx]:.3f}s")
        print(f"[DEBUG]   Landing: {t[landing_idx]:.3f}s")
    
    return phases_dict




def compute_phase_metrics(df: pd.DataFrame, phases: dict, bw: float, fs: int):
    """
    Compute comprehensive metrics for each CMJ phase + overall jump metrics.
    
    Returns dict with metrics for unweighting, braking, propulsive, and overall jump.
    """
    if not phases or len(phases) == 0:
        return {"unw": {}, "brk": {}, "prop": {}, "overall": {}}
    
    force = df["Z Total"].to_numpy()
    vel = df["Vel"].to_numpy()
    time = df["Time"].to_numpy()
    
    # Get left/right leg data if available
    has_legs = "Z Left" in df.columns and "Z Right" in df.columns
    if has_legs:
        force_left = df["Z Left"].to_numpy()
        force_right = df["Z Right"].to_numpy()
    
    g = 9.81
    mass = bw / g
    dt = 1 / fs
    
    metrics = {"unw": {}, "brk": {}, "prop": {}, "overall": {}}
    
    # Helper function to calculate asymmetry (Right = 100%, Left varies)
    def calc_asymmetry(left_val, right_val):
        if right_val == 0:
            return 0
        return ((left_val - right_val) / right_val) * 100
    
    # Helper function to get indices from time
    def time_to_idx(t):
        return np.argmin(np.abs(time - t))
    
    # ===== UNWEIGHTING PHASE =====
    if 'unweight_start' in phases and 'braking_start' in phases:
        start_idx = time_to_idx(phases['unweight_start'])
        end_idx = time_to_idx(phases['braking_start'])
        
        # Time
        unw_time = phases['braking_start'] - phases['unweight_start']
        
        # Force metrics
        unw_force_segment = force[start_idx:end_idx+1]
        unw_min_force = np.min(unw_force_segment)
        unw_min_force_pct = (unw_min_force / bw) * 100
        
        # Impulse (area under force-time curve)
        unw_impulse = np.trapz(unw_force_segment - bw, dx=dt)  # Net impulse
        
        # Leg-specific metrics
        if has_legs:
            unw_left_segment = force_left[start_idx:end_idx+1]
            unw_right_segment = force_right[start_idx:end_idx+1]
            
            unw_min_force_left = np.min(unw_left_segment)
            unw_min_force_right = np.min(unw_right_segment)
            unw_min_force_asym = calc_asymmetry(unw_min_force_left, unw_min_force_right)
            
            unw_impulse_left = np.trapz(unw_left_segment - (bw/2), dx=dt)
            unw_impulse_right = np.trapz(unw_right_segment - (bw/2), dx=dt)
            unw_impulse_asym = calc_asymmetry(unw_impulse_left, unw_impulse_right)
        else:
            unw_min_force_left = unw_min_force_right = unw_min_force_asym = None
            unw_impulse_left = unw_impulse_right = unw_impulse_asym = None
        
        metrics["unw"] = {
            "Time (s)": round(unw_time, 3),
            "Min Force (N)": round(unw_min_force, 1),
            "Min Force (% BW)": round(unw_min_force_pct, 1),
            "Impulse (N·s)": round(unw_impulse, 2),
            "Min Force Left (N)": round(unw_min_force_left, 1) if has_legs else "N/A",
            "Min Force Right (N)": round(unw_min_force_right, 1) if has_legs else "N/A",
            "Min Force Asym (%)": round(unw_min_force_asym, 1) if has_legs else "N/A",
            "Impulse Left (N·s)": round(unw_impulse_left, 2) if has_legs else "N/A",
            "Impulse Right (N·s)": round(unw_impulse_right, 2) if has_legs else "N/A",
            "Impulse Asym (%)": round(unw_impulse_asym, 1) if has_legs else "N/A"
        }
    
    # ===== BRAKING PHASE (Eccentric) =====
    if 'braking_start' in phases and 'propulsive_start' in phases:
        start_idx = time_to_idx(phases['braking_start'])
        end_idx = time_to_idx(phases['propulsive_start'])
        
        # Time
        brk_time = phases['propulsive_start'] - phases['braking_start']
        
        # Force metrics
        brk_force_segment = force[start_idx:end_idx+1]
        brk_max_force = np.max(brk_force_segment)
        brk_max_force_pct = (brk_max_force / bw) * 100
        brk_mean_force = np.mean(brk_force_segment)
        
        # Impulse
        brk_impulse = np.trapz(brk_force_segment - bw, dx=dt)
        
        # Velocity metrics (eccentric = negative velocity)
        brk_vel_segment = vel[start_idx:end_idx+1]
        brk_min_vel = np.min(brk_vel_segment)  # Most negative = peak eccentric velocity
        
        # Power metrics (Power = Force × Velocity)
        brk_power = brk_force_segment * brk_vel_segment
        brk_mean_power = np.mean(brk_power)
        brk_peak_power = np.min(brk_power)  # Most negative = peak eccentric power
        
        # Leg-specific metrics
        if has_legs:
            brk_left_segment = force_left[start_idx:end_idx+1]
            brk_right_segment = force_right[start_idx:end_idx+1]
            
            brk_max_force_left = np.max(brk_left_segment)
            brk_max_force_right = np.max(brk_right_segment)
            brk_max_force_asym = calc_asymmetry(brk_max_force_left, brk_max_force_right)
            
            brk_impulse_left = np.trapz(brk_left_segment - (bw/2), dx=dt)
            brk_impulse_right = np.trapz(brk_right_segment - (bw/2), dx=dt)
            brk_impulse_asym = calc_asymmetry(brk_impulse_left, brk_impulse_right)
        else:
            brk_max_force_left = brk_max_force_right = brk_max_force_asym = None
            brk_impulse_left = brk_impulse_right = brk_impulse_asym = None
        
        metrics["brk"] = {
            "Time (s)": round(brk_time, 3),
            "Max Force (N)": round(brk_max_force, 1),
            "Max Force (% BW)": round(brk_max_force_pct, 1),
            "Mean Force (N)": round(brk_mean_force, 1),
            "Min Velocity (m/s)": round(brk_min_vel, 3),
            "Impulse (N·s)": round(brk_impulse, 2),
            "Mean Power (W)": round(brk_mean_power, 1),
            "Peak Power (W)": round(brk_peak_power, 1),
            "Max Force Left (N)": round(brk_max_force_left, 1) if has_legs else "N/A",
            "Max Force Right (N)": round(brk_max_force_right, 1) if has_legs else "N/A",
            "Max Force Asym (%)": round(brk_max_force_asym, 1) if has_legs else "N/A",
            "Impulse Left (N·s)": round(brk_impulse_left, 2) if has_legs else "N/A",
            "Impulse Right (N·s)": round(brk_impulse_right, 2) if has_legs else "N/A",
            "Impulse Asym (%)": round(brk_impulse_asym, 1) if has_legs else "N/A"
        }
    
    # ===== PROPULSIVE PHASE (Concentric) =====
    if 'propulsive_start' in phases and 'takeoff' in phases:
        start_idx = time_to_idx(phases['propulsive_start'])
        end_idx = time_to_idx(phases['takeoff'])
        
        # Time
        prop_time = phases['takeoff'] - phases['propulsive_start']
        
        # Force metrics
        prop_force_segment = force[start_idx:end_idx+1]
        prop_max_force = np.max(prop_force_segment)
        prop_max_force_pct = (prop_max_force / bw) * 100
        prop_mean_force = np.mean(prop_force_segment)
        
        # Impulse
        prop_impulse = np.trapz(prop_force_segment - bw, dx=dt)
        
        # Velocity metrics (concentric = positive velocity)
        prop_vel_segment = vel[start_idx:end_idx+1]
        prop_max_vel = np.max(prop_vel_segment)  # Peak concentric velocity
        prop_takeoff_vel = vel[end_idx]  # Velocity at takeoff
        
        # Power metrics (Power = Force × Velocity)
        prop_power = prop_force_segment * prop_vel_segment
        prop_mean_power = np.mean(prop_power)
        prop_peak_power = np.max(prop_power)  # Peak concentric power
        
        # Leg-specific metrics
        if has_legs:
            prop_left_segment = force_left[start_idx:end_idx+1]
            prop_right_segment = force_right[start_idx:end_idx+1]
            
            prop_max_force_left = np.max(prop_left_segment)
            prop_max_force_right = np.max(prop_right_segment)
            prop_max_force_asym = calc_asymmetry(prop_max_force_left, prop_max_force_right)
            
            prop_impulse_left = np.trapz(prop_left_segment - (bw/2), dx=dt)
            prop_impulse_right = np.trapz(prop_right_segment - (bw/2), dx=dt)
            prop_impulse_asym = calc_asymmetry(prop_impulse_left, prop_impulse_right)
            
            # Power asymmetry
            prop_power_left = prop_left_segment * prop_vel_segment
            prop_power_right = prop_right_segment * prop_vel_segment
            prop_mean_power_left = np.mean(prop_power_left)
            prop_mean_power_right = np.mean(prop_power_right)
            prop_mean_power_asym = calc_asymmetry(prop_mean_power_left, prop_mean_power_right)
            
            prop_peak_power_left = np.max(prop_power_left)
            prop_peak_power_right = np.max(prop_power_right)
            prop_peak_power_asym = calc_asymmetry(prop_peak_power_left, prop_peak_power_right)
        else:
            prop_max_force_left = prop_max_force_right = prop_max_force_asym = None
            prop_impulse_left = prop_impulse_right = prop_impulse_asym = None
            prop_mean_power_left = prop_mean_power_right = prop_mean_power_asym = None
            prop_peak_power_left = prop_peak_power_right = prop_peak_power_asym = None
        
        metrics["prop"] = {
            "Time (s)": round(prop_time, 3),
            "Max Force (N)": round(prop_max_force, 1),
            "Max Force (% BW)": round(prop_max_force_pct, 1),
            "Mean Force (N)": round(prop_mean_force, 1),
            "Max Velocity (m/s)": round(prop_max_vel, 3),
            "Takeoff Velocity (m/s)": round(prop_takeoff_vel, 3),
            "Impulse (N·s)": round(prop_impulse, 2),
            "Mean Power (W)": round(prop_mean_power, 1),
            "Peak Power (W)": round(prop_peak_power, 1),
            "Max Force Left (N)": round(prop_max_force_left, 1) if has_legs else "N/A",
            "Max Force Right (N)": round(prop_max_force_right, 1) if has_legs else "N/A",
            "Max Force Asym (%)": round(prop_max_force_asym, 1) if has_legs else "N/A",
            "Impulse Left (N·s)": round(prop_impulse_left, 2) if has_legs else "N/A",
            "Impulse Right (N·s)": round(prop_impulse_right, 2) if has_legs else "N/A",
            "Impulse Asym (%)": round(prop_impulse_asym, 1) if has_legs else "N/A",
            "Mean Power Left (W)": round(prop_mean_power_left, 1) if has_legs else "N/A",
            "Mean Power Right (W)": round(prop_mean_power_right, 1) if has_legs else "N/A",
            "Mean Power Asym (%)": round(prop_mean_power_asym, 1) if has_legs else "N/A",
            "Peak Power Left (W)": round(prop_peak_power_left, 1) if has_legs else "N/A",
            "Peak Power Right (W)": round(prop_peak_power_right, 1) if has_legs else "N/A",
            "Peak Power Asym (%)": round(prop_peak_power_asym, 1) if has_legs else "N/A"
        }
    
    # ===== OVERALL JUMP METRICS =====
    if 'unweight_start' in phases and 'takeoff' in phases and 'propulsive_start' in phases:
        # Contraction time (total movement time from unweight start to takeoff)
        contraction_time = phases['takeoff'] - phases['unweight_start']
        
        # Jump height from impulse-momentum
        # Method 1: From takeoff velocity
        takeoff_idx = time_to_idx(phases['takeoff'])
        takeoff_velocity = vel[takeoff_idx]
        jump_height_vel = (takeoff_velocity ** 2) / (2 * g)
        
        # Method 2: From propulsive impulse
        prop_start_idx = time_to_idx(phases['propulsive_start'])
        takeoff_idx = time_to_idx(phases['takeoff'])
        prop_force_full = force[prop_start_idx:takeoff_idx+1]
        prop_impulse_full = np.trapz(prop_force_full - bw, dx=dt)
        
        # Change in velocity from impulse: Δv = J/m
        delta_v = prop_impulse_full / mass
        jump_height_impulse = (delta_v ** 2) / (2 * g)
        
        # Use average of both methods
        jump_height = (jump_height_vel + jump_height_impulse) / 2
        
        # RSI-modified (Jump Height / Contraction Time)
        rsi_mod = jump_height / contraction_time if contraction_time > 0 else 0
        
        # Overall power metrics (entire movement)
        movement_start_idx = time_to_idx(phases['unweight_start'])
        movement_force = force[movement_start_idx:takeoff_idx+1]
        movement_vel = vel[movement_start_idx:takeoff_idx+1]
        movement_power = movement_force * movement_vel
        
        mean_power_total = np.mean(movement_power)
        peak_power_total = np.max(movement_power)
        
        # Relative power (per kg body mass)
        mean_power_rel = mean_power_total / mass
        peak_power_rel = peak_power_total / mass
        
        # ===== ADD CMJ DEPTH HERE ===== ← RIGHT HERE!
        # CMJ DEPTH calculation
        eccentric_start_idx = time_to_idx(phases['unweight_start'])
        propulsive_start_idx = time_to_idx(phases['propulsive_start'])
        
        eccentric_vel = vel[eccentric_start_idx:propulsive_start_idx+1]
        
        if len(eccentric_vel) > 0:
            # Integrate velocity to get displacement (depth)
            cmj_depth = -np.trapz(eccentric_vel, dx=dt)  # Negate because velocity is negative
            cmj_depth_cm = cmj_depth * 100  # Convert to cm
        else:
            cmj_depth = 0
            cmj_depth_cm = 0
        # ===== END CMJ DEPTH =====
        
        metrics["overall"] = {
            "Contraction Time (s)": round(contraction_time, 3),
            "Jump Height (m)": round(jump_height, 3),
            "Jump Height (cm)": round(jump_height * 100, 1),
            "CMJ Depth (m)": round(cmj_depth, 3),           # ← ADD THIS LINE
            "CMJ Depth (cm)": round(cmj_depth_cm, 1),       # ← ADD THIS LINE
            "Takeoff Velocity (m/s)": round(takeoff_velocity, 3),
            "RSI-modified": round(rsi_mod, 3),
            "Mean Power (W)": round(mean_power_total, 1),
            "Peak Power (W)": round(peak_power_total, 1),
            "Mean Power (W/kg)": round(mean_power_rel, 1),
            "Peak Power (W/kg)": round(peak_power_rel, 1)
        }

    return metrics