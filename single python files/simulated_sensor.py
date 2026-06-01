"""
simulated_sensor.py
===================
Simulates a real-time ShieldNode sensor stream using IoTsec or EPW data.

Replays historical data through the AlertStateMachine at configurable
speed, printing live state updates as if reading from a real sensor.

Usage:
    # Replay IoTsec location A, measurement 10 at 10x speed
    python simulated_sensor.py --source iotsec --location A --measurement 10

    # Replay Miami EPW at 100x speed
    python simulated_sensor.py --source epw --city miami

    # Run a worst-case synthetic scenario
    python simulated_sensor.py --source synthetic --scenario worst_case
"""

import argparse
import time
import sys
import csv
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Import alert state machine
sys.path.insert(0, str(Path(__file__).parent))
from alert_state import AlertStateMachine, AlertLevel

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE      = Path.home() / "Desktop/mold_risk_model"
IOTSEC    = BASE / "data/raw/iotsec"
EPW_DIR   = BASE / "data/raw/epw"

# ── ANSI colors for terminal output ──────────────────────────────────────────

COLORS = {
    "SAFE":     "\033[92m",   # green
    "MONITOR":  "\033[93m",   # yellow
    "WARNING":  "\033[91m",   # red
    "CRITICAL": "\033[95m",   # magenta
    "RESET":    "\033[0m",
    "BOLD":     "\033[1m",
    "DIM":      "\033[2m",
}


def colorize(text: str, level_name: str) -> str:
    color = COLORS.get(level_name, "")
    return f"{color}{text}{COLORS['RESET']}"


def print_header():
    print(f"\n{COLORS['BOLD']}{'='*62}")
    print("  ShieldNode — Simulated Sensor Stream")
    print(f"{'='*62}{COLORS['RESET']}\n")


def print_state(state: dict, reading_num: int,
                sim_time: datetime, speed: float):
    level = state["level_name"]
    bar   = "█" * int(state["M"] / 6 * 20)
    bar  += "░" * (20 - len(bar))

    line = (
        f"  [{sim_time.strftime('%m/%d %H:%M')}]  "
        f"M={state['M']:5.3f} [{bar}]  "
        f"RH={state['rh_surface']:5.1f}%  "
        f"T_crit={state['rh_crit']:5.1f}%  "
    )
    alert_tag = colorize(f"[{level:<8}]", level)
    print(f"\r{line}{alert_tag}", end="", flush=True)

    # Print full message on level change or every 24 sim-hours
    if reading_num % (24 * 60) == 0 or reading_num == 1:
        print()
        print(colorize(f"    → {state['message']}", level))


def print_summary(asm: AlertStateMachine,
                  level_counts: dict, total: int):
    print(f"\n\n{COLORS['BOLD']}{'─'*62}")
    print("  Session Summary")
    print(f"{'─'*62}{COLORS['RESET']}")
    s = asm.summary
    print(f"  Material   : {s['material']}")
    print(f"  Duration   : {s['total_days']:.1f} days "
          f"({s['total_hours']:.0f} hours)")
    print(f"  Final M    : {s['M']:.3f}/6")
    print(f"  Final level: "
          + colorize(s['current_level'], s['current_level']))
    print()
    for level, count in sorted(level_counts.items(),
                                key=lambda x: x[0].value):
        pct  = count / total * 100
        bar  = "█" * int(pct / 2)
        name = colorize(f"{level.name:<8}", level.name)
        print(f"  {name}  {pct:5.1f}%  {bar}")
    print(f"{'─'*62}\n")


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_iotsec(location: str, measurement: int) -> pd.DataFrame:
    """Load one IoTsec measurement. Returns df with temp/rh columns."""
    path = IOTSEC / f"location_{location}" / f"measurement{measurement:02d}.csv"
    if not path.exists():
        raise FileNotFoundError(f"IoTsec file not found: {path}")

    cols = ["EID","AbsT","RelT","NID","Temp","RelH",
            "L1","L2","Occ","Act","Door","Win"]
    df = pd.read_csv(path, header=None, names=cols)

    # Average across nodes per timestamp
    df = (df.groupby("RelT")[["Temp","RelH"]]
            .mean()
            .reset_index()
            .sort_values("RelT"))
    df["dt_hours"] = 1/60  # ~1 min intervals
    return df


def load_epw(city_key: str) -> pd.DataFrame:
    """Load EPW file and return hourly temp/rh DataFrame."""
    matches = list(EPW_DIR.glob(f"*{city_key}*.epw"))
    if not matches:
        raise FileNotFoundError(
            f"No EPW file found for '{city_key}' in {EPW_DIR}"
        )
    path = matches[0]
    print(f"  Loading EPW: {path.name}")

    epw_cols = [
        "year","month","day","hour","minute","data_source",
        "dry_bulb_temp","dew_point_temp","rel_humidity",
        "atm_pressure","extraterr_horiz_rad","extraterr_direct_rad",
        "horiz_infrared_rad","global_horiz_rad","direct_normal_rad",
        "diffuse_horiz_rad","global_horiz_illum","direct_normal_illum",
        "diffuse_horiz_illum","zenith_luminance","wind_direction",
        "wind_speed","total_sky_cover","opaque_sky_cover","visibility",
        "ceiling_height","present_weather_obs","present_weather_codes",
        "precip_water","aerosol_opt_depth","snow_depth","days_since_snow",
        "albedo","liquid_precip_depth","liquid_precip_rate"
    ]
    df = pd.read_csv(path, skiprows=8, header=None,
                     names=epw_cols, usecols=range(len(epw_cols)))

    # Compute cavity temp (R-13, AC at 22°C)
    alpha    = 0.75
    out_temp = df["dry_bulb_temp"].values.astype(float)
    out_rh   = df["rel_humidity"].values.astype(float)
    cav_temp = out_temp + alpha * (22.0 - out_temp)

    # Cavity RH via Magnus
    def e_sat(t):
        return 6.1078 * np.exp(17.27 * t / (t + 237.3))

    e_out   = (out_rh / 100.0) * e_sat(out_temp)
    cav_rh  = np.clip(e_out / e_sat(cav_temp), 0, 1) * 100.0

    result = pd.DataFrame({
        "Temp":     cav_temp,
        "RelH":     cav_rh,
        "dt_hours": 1.0,
    })
    return result


def generate_synthetic(scenario: str) -> pd.DataFrame:
    """Generate a synthetic sensor stream for a named scenario."""
    scenarios = {
        "safe": {
            "desc": "Stable safe indoor conditions",
            "temp": 22.0, "rh": 55.0, "hours": 30 * 24
        },
        "worst_case": {
            "desc": "Hot humid wall cavity — Miami summer",
            "temp": 20.0, "rh": 92.0, "hours": 90 * 24
        },
        "seasonal": {
            "desc": "Seasonal cycle — safe winter, risky summer",
        },
        "recovery": {
            "desc": "Dangerous conditions followed by remediation",
        },
    }

    if scenario not in scenarios:
        raise ValueError(
            f"Unknown scenario '{scenario}'. "
            f"Choose from: {list(scenarios.keys())}"
        )

    print(f"  Scenario: {scenarios[scenario]['desc']}")

    if scenario in ("safe", "worst_case"):
        s     = scenarios[scenario]
        n     = s["hours"]
        temps = np.full(n, s["temp"])
        rhs   = np.full(n, s["rh"])

    elif scenario == "seasonal":
        n     = 365 * 24
        hours = np.arange(n)
        # Temperature: 15°C winter to 25°C summer
        temps = 20 + 5 * np.sin(2 * np.pi * (hours - 2160) / 8760)
        # RH: 55% winter to 90% summer with diurnal variation
        rhs   = (72 + 18 * np.sin(2 * np.pi * (hours - 2160) / 8760)
                 + 8  * np.sin(2 * np.pi * hours / 24))
        rhs   = np.clip(rhs, 30, 100)

    elif scenario == "recovery":
        # 60 days dangerous, then dehumidifier kicks in
        n_danger   = 60 * 24
        n_recovery = 30 * 24
        temps = np.concatenate([
            np.full(n_danger,   18.0),
            np.full(n_recovery, 22.0),
        ])
        rhs = np.concatenate([
            np.full(n_danger,   92.0),
            np.full(n_recovery, 50.0),
        ])

    return pd.DataFrame({
        "Temp":     temps,
        "RelH":     rhs,
        "dt_hours": 1.0,
    })


# ── Main simulation loop ──────────────────────────────────────────────────────

def run_simulation(df: pd.DataFrame,
                   material:    str   = "sensitive",
                   speed:       float = 100.0,
                   start_dt:    datetime = None,
                   log_csv:     str   = None,
                   scenario_tag: str  = "run"):
    """
    Replay sensor data through AlertStateMachine.

    Parameters
    ----------
    df        : DataFrame with columns Temp, RelH, dt_hours
    material  : material sensitivity class
    speed     : playback speed multiplier (1.0 = real time)
    start_dt  : simulation start datetime
    log_csv   : optional path to write CSV log
    """
    if start_dt is None:
        start_dt = datetime(2024, 1, 1, 0, 0)

    asm          = AlertStateMachine(material=material)
    level_counts = {lvl: 0 for lvl in AlertLevel}
    sim_time     = start_dt
    log_rows     = []

    print_header()
    print(f"  Material : {material}")
    print(f"  Readings : {len(df):,}")
    print(f"  Speed    : {speed}x  "
          f"({'real-time' if speed == 1 else 'simulated'})\n")

    prev_level = AlertLevel.SAFE
    last_print = 0

    try:
        for i, row in enumerate(df.itertuples(index=False)):
            dt   = float(row.dt_hours)
            asm.dt_hours = dt

            state = asm.update(
                cavity_temp_c=float(row.Temp),
                cavity_rh_pct=float(row.RelH),
            )

            level_counts[state["level"]] += 1
            sim_time += timedelta(hours=dt)

            # Print on level change
            if state["level"] != prev_level:
                print()
                tag = colorize(
                    f"  ▶ LEVEL CHANGE: "
                    f"{prev_level.name} → {state['level_name']}",
                    state["level_name"]
                )
                print(tag)
                print(f"    {state['message']}")
                print(f"    M={state['M']:.3f}  "
                      f"RH={state['rh_surface']:.1f}%  "
                      f"Day {state['total_hours']/24:.1f}")
                prev_level = state["level"]

            # Periodic status line (every simulated hour)
            if i - last_print >= int(1 / dt):
                print_state(state, i+1, sim_time, speed)
                last_print = i

            # Log row
            if log_csv:
                log_rows.append({
                    "sim_time":    sim_time.isoformat(),
                    "temp_c":      row.Temp,
                    "rh_pct":      row.RelH,
                    "M":           state["M"],
                    "level":       state["level_name"],
                    "rh_surface":  state["rh_surface"],
                    "rh_crit":     state["rh_crit"],
                    "exceedance":  state["exceedance"],
                })

                

            # Throttle for real-time feel
            if speed < 9999:
                time.sleep(dt / speed)

    except KeyboardInterrupt:
        print("\n\n  [Stopped by user]")

    print_summary(asm, level_counts, max(1, i+1))

    # Auto-save JSON summary to pyresults
    import json
    from pathlib import Path
    out_dir = Path(__file__).parent / "pyresults"
    out_dir.mkdir(exist_ok=True)

    summary_out = {
        "source":       "simulated_sensor",
        "material":     asm.material,
        "total_days":   round(asm.total_hours / 24, 1),
        "total_hours":  round(asm.total_hours, 1),
        "final_M":      round(float(asm.M), 4),
        "final_level":  asm.current_level.name,
        "level_pct": {
            lvl.name: round(count / max(1, i+1) * 100, 1)
            for lvl, count in level_counts.items()
        },
    }
    tag      = getattr(args, 'scenario', 'run') \
               if 'args' in dir() else 'run'
    out_path = out_dir / f"sim_{scenario_tag}_{asm.material}.json"
    with open(out_path, "w") as f:
        json.dump(summary_out, f, indent=2)
    print(f"  Summary saved: {out_path}")

    # Write CSV log
    if log_csv and log_rows:
        out = Path(log_csv)
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"  Log saved: {out}")

    return asm


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="ShieldNode simulated sensor stream"
    )
    p.add_argument("--source", choices=["iotsec","epw","synthetic"],
                   default="synthetic",
                   help="Data source (default: synthetic)")
    p.add_argument("--location", default="A",
                   help="IoTsec location A/B/C (iotsec source)")
    p.add_argument("--measurement", type=int, default=10,
                   help="IoTsec measurement number (iotsec source)")
    p.add_argument("--city", default="Miami",
                   help="City substring to match EPW file (epw source)")
    p.add_argument("--scenario", default="seasonal",
                   choices=["safe","worst_case","seasonal","recovery"],
                   help="Synthetic scenario (synthetic source)")
    p.add_argument("--material", default="sensitive",
                   choices=["very_sensitive","sensitive",
                            "medium_resistant","resistant"],
                   help="Material sensitivity class")
    p.add_argument("--speed", type=float, default=500.0,
                   help="Playback speed multiplier (default 500x)")
    p.add_argument("--log", default=None,
                   help="Optional path to write CSV log")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    print(f"\nSource: {args.source}")

    if args.source == "iotsec":
        df = load_iotsec(args.location, args.measurement)
        print(f"  Loaded IoTsec location {args.location} "
              f"measurement {args.measurement:02d} "
              f"({len(df)} readings)")
    elif args.source == "epw":
        df = load_epw(args.city)
    else:
        df = generate_synthetic(args.scenario)

    run_simulation(
        df,
        material=args.material,
        speed=args.speed,
        log_csv=args.log,
        scenario_tag=args.scenario,
    )