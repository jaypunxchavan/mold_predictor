"""
benchmark.py
============
Benchmarks ShieldNode's VTT mold model against a naive RH threshold
detector on EPW climate data.

Answers the judge question: "Why not just use a cheap hygrometer?"

Metrics compared:
  - Alert lead time (days before M=3 visual threshold)
  - False positive rate (alerts when no real risk)
  - False negative rate (missed risk events)
  - Alert stability (how often the alert toggles)

Usage:
    python benchmark.py
    python benchmark.py --city Miami --rh_thresh 70
    python benchmark.py --all_cities
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from alert_state import AlertStateMachine, AlertLevel, MATERIAL_PARAMS

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE    = Path(__file__).parent.parent
EPW_DIR = BASE / "data/raw/epw"
OUT_DIR = Path(__file__).parent / "pyresults"
OUT_DIR.mkdir(exist_ok=True)

CITY_LABELS = {
    'USA_CA_Port.Chicago.': 'Port Chicago, CA',
    'USA_FL_Miami.Intl.AP': 'Miami, FL',
    'USA_LA_New.Orleans-N': 'New Orleans, LA',
    'USA_TX_Houston-Bush.': 'Houston, TX',
}


# ── EPW loader ────────────────────────────────────────────────────────────────

def load_epw(path: Path) -> pd.DataFrame:
    cols = [
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
                     names=cols, usecols=range(len(cols)))

    # Derive cavity conditions (R-13, AC at 22°C)
    alpha    = 0.75
    out_temp = df["dry_bulb_temp"].values.astype(float)
    out_rh   = df["rel_humidity"].values.astype(float)
    cav_temp = out_temp + alpha * (22.0 - out_temp)

    def e_sat(t):
        return 6.1078 * np.exp(17.27 * t / (t + 237.3))

    e_out   = (out_rh / 100.0) * e_sat(out_temp)
    cav_rh  = np.clip(e_out / e_sat(cav_temp), 0, 1) * 100.0

    return pd.DataFrame({
        "hour":       np.arange(len(df)),
        "out_temp":   out_temp,
        "out_rh":     out_rh,
        "cav_temp":   cav_temp,
        "cav_rh":     cav_rh,
    })


# ── Naive threshold detector ──────────────────────────────────────────────────

def run_naive_threshold(df: pd.DataFrame,
                        rh_thresh:      float = 70.0,
                        debounce_hours: int   = 2) -> np.ndarray:
    """
    Simple RH threshold detector — what a cheap hygrometer does.

    Fires alert when outdoor RH > rh_thresh for debounce_hours
    consecutive hours. Uses outdoor RH (what the device measures),
    not cavity RH (what actually matters).

    Returns boolean alert array (True = alert active).
    """
    rh      = df["out_rh"].values
    above   = rh > rh_thresh
    alerts  = np.zeros(len(rh), dtype=bool)
    counter = 0

    for i, a in enumerate(above):
        if a:
            counter += 1
            if counter >= debounce_hours:
                alerts[i] = True
        else:
            counter = 0

    return alerts


# ── VTT detector ──────────────────────────────────────────────────────────────

def run_vtt_detector(df: pd.DataFrame,
                     material: str = "very_sensitive") -> tuple:
    """
    Run VTT model through AlertStateMachine on EPW cavity data.

    Returns:
        alerts_1 : bool array, M >= 1 (MONITOR level)
        alerts_3 : bool array, M >= 3 (WARNING level)
        M_series : float array, full mold index time series
    """
    asm      = AlertStateMachine(material=material, dt_hours=1.0)
    n        = len(df)
    M_series = np.zeros(n)
    alerts_1 = np.zeros(n, dtype=bool)
    alerts_3 = np.zeros(n, dtype=bool)

    for i, row in enumerate(df.itertuples(index=False)):
        asm.dt_hours = 1.0
        state        = asm.update(
            cavity_temp_c=float(row.cav_temp),
            cavity_rh_pct=float(row.cav_rh),
        )
        M_series[i] = state["M"]
        alerts_1[i] = state["M"] >= 1.0
        alerts_3[i] = state["M"] >= 3.0

    return alerts_1, alerts_3, M_series


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(naive_alerts: np.ndarray,
                    vtt_alerts_1: np.ndarray,
                    vtt_alerts_3: np.ndarray,
                    M_series:     np.ndarray,
                    city_label:   str,
                    rh_thresh:    float) -> dict:
    """
    Compare naive threshold vs VTT model.

    Ground truth: VTT M >= 3 (visually detectable) = true mold event.
    """
    n          = len(naive_alerts)
    true_risk  = vtt_alerts_3        # ground truth
    no_risk    = ~true_risk

    # ── Naive metrics ──────────────────────────────────────────────
    naive_tp   = (naive_alerts &  true_risk).sum()
    naive_fp   = (naive_alerts &  no_risk).sum()
    naive_fn   = (~naive_alerts & true_risk).sum()
    naive_tn   = (~naive_alerts & no_risk).sum()

    naive_fpr  = naive_fp / max(1, no_risk.sum())    * 100
    naive_fnr  = naive_fn / max(1, true_risk.sum())  * 100
    naive_prec = naive_tp / max(1, naive_alerts.sum()) * 100

    # Alert toggles (false stability — annoying for property manager)
    naive_toggles = int(np.diff(naive_alerts.astype(int)).abs().sum()
                        if hasattr(np.diff(naive_alerts.astype(int)), 'abs')
                        else np.abs(np.diff(naive_alerts.astype(int))).sum())

    # ── VTT metrics ────────────────────────────────────────────────
    vtt_tp   = (vtt_alerts_1 &  true_risk).sum()
    vtt_fp   = (vtt_alerts_1 &  no_risk).sum()
    vtt_fn   = (~vtt_alerts_1 & true_risk).sum()

    vtt_fpr  = vtt_fp / max(1, no_risk.sum())    * 100
    vtt_fnr  = vtt_fn / max(1, true_risk.sum())  * 100
    vtt_prec = vtt_tp / max(1, vtt_alerts_1.sum()) * 100

    vtt_toggles = int(np.abs(np.diff(vtt_alerts_1.astype(int))).sum())

    # ── Lead time ──────────────────────────────────────────────────
    # How many hours does VTT fire BEFORE naive first fires
    # during the first true risk episode?
    lead_time_hours = None
    lead_time_days  = None

    if true_risk.any():
        first_true = int(np.argmax(true_risk))

        # First VTT alert before the true risk event
        vtt_before = vtt_alerts_1[:first_true]
        if vtt_before.any():
            first_vtt = int(np.argmax(vtt_before))
        else:
            first_vtt = first_true

        # First naive alert before or at true risk
        naive_before = naive_alerts[:first_true + 1]
        if naive_before.any():
            first_naive = int(np.argmax(naive_before))
        else:
            first_naive = first_true

        lead_time_hours = first_naive - first_vtt
        lead_time_days  = round(lead_time_hours / 24, 1)

    # ── Pct time alerting ──────────────────────────────────────────
    naive_pct = naive_alerts.mean() * 100
    vtt_pct   = vtt_alerts_1.mean() * 100
    true_pct  = true_risk.mean()    * 100
    max_M     = float(M_series.max())

    return {
        "city":             city_label,
        "rh_threshold":     rh_thresh,
        "n_hours":          n,
        "true_risk_pct":    round(true_pct, 1),
        "max_M":            round(max_M, 3),

        "naive": {
            "alert_pct":    round(naive_pct, 1),
            "false_pos_rate": round(naive_fpr, 1),
            "false_neg_rate": round(naive_fnr, 1),
            "precision":    round(naive_prec, 1),
            "toggles":      naive_toggles,
        },
        "vtt": {
            "alert_pct":    round(vtt_pct, 1),
            "false_pos_rate": round(vtt_fpr, 1),
            "false_neg_rate": round(vtt_fnr, 1),
            "precision":    round(vtt_prec, 1),
            "toggles":      vtt_toggles,
        },
        "lead_time_hours":  lead_time_hours,
        "lead_time_days":   lead_time_days,
    }


# ── Report printer ────────────────────────────────────────────────────────────

def print_report(metrics: dict):
    c = metrics
    w = 62

    print(f"\n{'═'*w}")
    print(f"  {c['city']}  ·  RH threshold: {c['rh_threshold']}%")
    print(f"{'─'*w}")
    print(f"  Ground truth (VTT M≥3) : {c['true_risk_pct']:5.1f}% of year")
    print(f"  Max mold index         : {c['max_M']:.3f} / 6")
    print(f"{'─'*w}")
    print(f"  {'Metric':<28} {'Naive RH':>10} {'ShieldNode':>10}")
    print(f"  {'─'*28} {'─'*10} {'─'*10}")

    rows = [
        ("% year in alert",       c['naive']['alert_pct'],
                                  c['vtt']['alert_pct'],     "%"),
        ("False positive rate",   c['naive']['false_pos_rate'],
                                  c['vtt']['false_pos_rate'], "%"),
        ("False negative rate",   c['naive']['false_neg_rate'],
                                  c['vtt']['false_neg_rate'], "%"),
        ("Precision",             c['naive']['precision'],
                                  c['vtt']['precision'],      "%"),
        ("Alert toggles/year",    c['naive']['toggles'],
                                  c['vtt']['toggles'],        ""),
    ]

    for label, naive_val, vtt_val, unit in rows:
        # Highlight which is better
        if label in ("False positive rate", "False negative rate",
                     "Alert toggles/year"):
            better = "vtt" if vtt_val < naive_val else "naive"
        else:
            better = "vtt" if vtt_val > naive_val else "naive"

        naive_str = f"{naive_val:.1f}{unit}"
        vtt_str   = f"{vtt_val:.1f}{unit}"

        if better == "vtt":
            vtt_str   = f"✓ {vtt_str}"
        else:
            naive_str = f"✓ {naive_str}"

        print(f"  {label:<28} {naive_str:>10} {vtt_str:>12}")

    if c['lead_time_days'] is not None:
        print(f"{'─'*w}")
        sign = "+" if c['lead_time_days'] > 0 else ""
        print(f"  ShieldNode lead time advantage : "
              f"{sign}{c['lead_time_days']} days "
              f"({c['lead_time_hours']:+d} hours)")
        if c['lead_time_days'] > 0:
            print(f"  → ShieldNode fires {c['lead_time_days']} days "
                  f"BEFORE naive threshold would alert")
        elif c['lead_time_days'] < 0:
            print(f"  → Naive threshold fires {abs(c['lead_time_days'])} days "
                  f"earlier (but with more false positives)")
        else:
            print(f"  → Both systems alert at the same time")

    print(f"{'═'*w}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_benchmark(epw_path:   Path,
                  rh_thresh:  float = 70.0,
                  material:   str   = "very_sensitive",
                  debounce:   int   = 2) -> dict:

    city_key   = epw_path.stem[:20]
    city_label = CITY_LABELS.get(city_key, epw_path.stem[:30])

    print(f"\nLoading {city_label}...")
    df = load_epw(epw_path)

    print(f"  Running naive RH>{rh_thresh}% detector...")
    naive_alerts = run_naive_threshold(df, rh_thresh, debounce)

    print(f"  Running VTT model ({material})...")
    vtt_1, vtt_3, M_series = run_vtt_detector(df, material)

    metrics = compute_metrics(
        naive_alerts, vtt_1, vtt_3, M_series,
        city_label, rh_thresh
    )
    print_report(metrics)
    return metrics


def main():
    p = argparse.ArgumentParser(
        description="Benchmark ShieldNode VTT vs naive RH threshold"
    )
    p.add_argument("--city", default="Miami",
                   help="City substring to match EPW file")
    p.add_argument("--rh_thresh", type=float, default=70.0,
                   help="Naive RH threshold percent(default 70)")
    p.add_argument("--material", default="very_sensitive",
                   choices=list(MATERIAL_PARAMS.keys()),
                   help="VTT material class")
    p.add_argument("--debounce", type=int, default=2,
                   help="Naive detector debounce hours (default 2)")
    p.add_argument("--all_cities", action="store_true",
                   help="Run benchmark for all available EPW files")
    args = p.parse_args()

    epw_files = sorted(EPW_DIR.glob("*.epw"))
    if not epw_files:
        print(f"No EPW files found in {EPW_DIR}")
        sys.exit(1)

    if args.all_cities:
        targets = epw_files
    else:
        targets = [f for f in epw_files
                   if args.city.lower() in f.name.lower()]
        if not targets:
            print(f"No EPW file matching '{args.city}' found.")
            print(f"Available: {[f.name for f in epw_files]}")
            sys.exit(1)

    all_metrics = []
    for epw_path in targets:
        m = run_benchmark(
            epw_path,
            rh_thresh=args.rh_thresh,
            material=args.material,
            debounce=args.debounce,
        )
        all_metrics.append(m)

    # Save results
    out_tag  = "all_cities" if args.all_cities else args.city.lower()
    out_path = OUT_DIR / f"benchmark_{out_tag}_rh{int(args.rh_thresh)}.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Results saved: {out_path}")

    # Print summary table for all_cities run
    if args.all_cities and len(all_metrics) > 1:
        print(f"\n{'─'*72}")
        print(f"  {'City':<25} {'Lead days':>10} {'Naive FPR':>10} "
              f"{'VTT FPR':>8} {'Naive FNR':>10} {'VTT FNR':>8}")
        print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*8} {'─'*10} {'─'*8}")
        for m in all_metrics:
            ld = f"{m['lead_time_days']:+.1f}d" \
                 if m['lead_time_days'] is not None else "n/a"
            print(f"  {m['city']:<25} {ld:>10} "
                  f"{m['naive']['false_pos_rate']:>9.1f}% "
                  f"{m['vtt']['false_pos_rate']:>7.1f}% "
                  f"{m['naive']['false_neg_rate']:>9.1f}% "
                  f"{m['vtt']['false_neg_rate']:>7.1f}%")
        print(f"{'─'*72}\n")


if __name__ == "__main__":
    main()