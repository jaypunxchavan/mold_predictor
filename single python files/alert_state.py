"""
alert_state.py
==============
Alert state machine for ShieldNode mold risk monitoring.

Wraps the VTT mold index with hysteresis, escalation levels,
and debounce logic to prevent false alert toggling.

Usage:
    from alert_state import AlertStateMachine
    
    asm = AlertStateMachine()
    for temp, rh in sensor_stream:
        state = asm.update(temp, rh)
        print(state)
"""

from enum import Enum
from collections import deque
import numpy as np
import json
from pathlib import Path


# ── Alert levels ──────────────────────────────────────────────────────────────

class AlertLevel(Enum):
    SAFE     = 0   # M < 1.0  — no biological activity
    MONITOR  = 1   # M >= 1.0 — microscopic growth beginning
    WARNING  = 2   # M >= 3.0 — visually detectable growth imminent
    CRITICAL = 3   # M >= 5.0 — heavy growth, immediate action required


# ── Thresholds ────────────────────────────────────────────────────────────────

ENTRY_THRESHOLDS = {
    AlertLevel.MONITOR:  1.0,
    AlertLevel.WARNING:  3.0,
    AlertLevel.CRITICAL: 5.0,
}

# Hysteresis: alert clears only when M drops below exit threshold
EXIT_THRESHOLDS = {
    AlertLevel.MONITOR:  0.80,
    AlertLevel.WARNING:  2.50,
    AlertLevel.CRITICAL: 4.50,
}

# Debounce: M must exceed entry threshold for this many consecutive
# hours before alert fires (prevents toggling on brief spikes)
DEBOUNCE_HOURS = {
    AlertLevel.MONITOR:  6,
    AlertLevel.WARNING:  3,
    AlertLevel.CRITICAL: 1,
}

MATERIAL_PARAMS = {
    "very_sensitive": {
        "k1_low": 1.000, "k1_high": 2.000,
        "rh_min": 80.0,
        "A": 1.0, "B": 7.0, "C": 2.0,
        "cmat": 1.0,
    },
    "sensitive": {
        "k1_low": 0.578, "k1_high": 0.386,
        "rh_min": 80.0,
        "A": 0.3, "B": 6.0, "C": 1.0,
        "cmat": 0.5,
    },
    "medium_resistant": {
        "k1_low": 0.072, "k1_high": 0.097,
        "rh_min": 85.0,
        "A": 0.0, "B": 5.0, "C": 1.5,
        "cmat": 0.25,
    },
    "resistant": {
        "k1_low": 0.033, "k1_high": 0.014,
        "rh_min": 85.0,
        "A": 0.0, "B": 3.0, "C": 1.0,
        "cmat": 0.1,
    },
}


# ── Core physics ──────────────────────────────────────────────────────────────

def rh_crit(temp_c: float, rh_min: float = 80.0) -> float:
    """VTT critical RH threshold. Ojanen et al. 2010, Eq. 1."""
    if temp_c <= 20:
        return (-0.00267 * temp_c**3
                + 0.160  * temp_c**2
                - 3.13   * temp_c
                + 100.0)
    return rh_min


def e_sat(t: float) -> float:
    """Magnus formula saturation vapor pressure (hPa)."""
    return 6.1078 * np.exp(17.27 * t / (t + 237.3))


def cavity_surface_rh(cavity_temp: float,
                       outdoor_temp: float,
                       outdoor_rh_pct: float) -> float:
    """RH at wall surface (%). Returns value 0–100."""
    e_out = (outdoor_rh_pct / 100.0) * e_sat(outdoor_temp)
    return float(np.clip(e_out / e_sat(cavity_temp), 0.0, 1.0) * 100.0)


# ── Alert State Machine ───────────────────────────────────────────────────────

class AlertStateMachine:
    """
    Real-time mold risk alert state machine.

    Maintains VTT mold index state across timesteps and applies
    debounce + hysteresis logic to produce stable alert levels.

    Parameters
    ----------
    material : str
        Material sensitivity class. Default 'sensitive' (paper-faced drywall).
    dt_hours : float
        Timestep in hours. Default 1/60 (one reading per minute).
    history_size : int
        Number of past readings to retain for diagnostics.
    """

    def __init__(self,
                 material:      str   = "sensitive",
                 dt_hours:      float = 1/60,
                 history_size:  int   = 1440):

        self.material     = material
        self.dt_hours     = dt_hours
        self.p            = MATERIAL_PARAMS[material]

        # VTT model state
        self.M            = 0.0
        self.rh_max       = 0.0
        self.t_unfav      = 0.0

        # Alert state
        self.current_level     = AlertLevel.SAFE
        self.debounce_counters = {lvl: 0 for lvl in AlertLevel}
        self.hours_in_state    = 0.0
        self.total_hours       = 0.0

        # History ring buffer
        self._history = deque(maxlen=history_size)

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self,
               cavity_temp_c:   float,
               cavity_rh_pct:   float,
               outdoor_temp_c:  float  = None,
               outdoor_rh_pct:  float  = None) -> dict:
        """
        Ingest one sensor reading and return current alert state.

        Parameters
        ----------
        cavity_temp_c  : wall cavity temperature (°C)
        cavity_rh_pct  : wall cavity RH (%) — direct measurement
                         OR pass outdoor_temp_c + outdoor_rh_pct
                         and set cavity_rh_pct=None to derive it.
        outdoor_temp_c : outdoor temperature (°C) — optional
        outdoor_rh_pct : outdoor RH (%) — optional

        Returns
        -------
        dict with keys:
            level        : AlertLevel enum
            level_name   : str
            M            : float, current mold index (0–6)
            rh_surface   : float, RH at cavity surface (%)
            rh_crit      : float, critical RH threshold (%)
            exceedance   : float, RH - RH_crit (%)
            hours_in_state : float
            message      : str, human-readable status
        """
        # Derive cavity RH from outdoor if direct not available
        if cavity_rh_pct is None:
            if outdoor_temp_c is None or outdoor_rh_pct is None:
                raise ValueError(
                    "Provide either cavity_rh_pct directly, "
                    "or both outdoor_temp_c and outdoor_rh_pct."
                )
            cavity_rh_pct = cavity_surface_rh(
                cavity_temp_c, outdoor_temp_c, outdoor_rh_pct
            )

        # Step VTT model
        self.M = self._step_vtt(cavity_temp_c, cavity_rh_pct)
        self.total_hours += self.dt_hours

        # Determine raw target level from M
        target = self._m_to_raw_level(self.M)

        # Apply debounce and hysteresis
        new_level = self._resolve_level(target)

        if new_level != self.current_level:
            self.hours_in_state = 0.0
        else:
            self.hours_in_state += self.dt_hours

        self.current_level = new_level

        rc      = rh_crit(cavity_temp_c, self.p["rh_min"])
        result  = {
            "level":         new_level,
            "level_name":    new_level.name,
            "M":             round(self.M, 4),
            "rh_surface":    round(cavity_rh_pct, 2),
            "rh_crit":       round(rc, 2),
            "exceedance":    round(cavity_rh_pct - rc, 2),
            "hours_in_state":round(self.hours_in_state, 2),
            "total_hours":   round(self.total_hours, 2),
            "message":       self._message(new_level),
        }

        self._history.append(result)
        return result

    def reset(self):
        """Reset all state — use when starting a new monitoring session."""
        self.M           = 0.0
        self.rh_max      = 0.0
        self.t_unfav     = 0.0
        self.current_level    = AlertLevel.SAFE
        self.debounce_counters = {lvl: 0 for lvl in AlertLevel}
        self.hours_in_state   = 0.0
        self.total_hours      = 0.0
        self._history.clear()

    @property
    def history(self) -> list:
        """Return list of past state dicts (up to history_size)."""
        return list(self._history)

    @property
    def summary(self) -> dict:
        """High-level summary of current monitoring status."""
        return {
            "material":      self.material,
            "current_level": self.current_level.name,
            "M":             round(self.M, 4),
            "total_hours":   round(self.total_hours, 2),
            "total_days":    round(self.total_hours / 24, 1),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _step_vtt(self, temp: float, rh: float) -> float:
        """Single-step VTT mold growth model. Returns updated M."""
        p      = self.p
        rh_cr  = rh_crit(temp, p["rh_min"])
        m_prev = self.M

        if rh > rh_cr and temp > 0:
            self.t_unfav = 0.0
            self.rh_max  = max(self.rh_max, rh)

            k1    = p["k1_low"] if m_prev < 1 else p["k1_high"]
            ratio = (rh_cr - self.rh_max) / (rh_cr - 100.0)
            m_max = max(0.0, p["A"] + p["B"] * ratio - p["C"] * ratio**2)
            k2    = max(1 - np.exp(2.3 * (m_prev - m_max)), 0) \
                    if m_max > 0 else 0.0

            denom = 7 * np.exp(
                -0.68 * np.log(max(temp, 0.1))
                - 13.9 * np.log(max(rh, 0.01))
                + 66.02
            )
            dm = (1.0 / denom) * k1 * k2 * self.dt_hours / 24.0
            return min(m_prev + dm, m_max)

        else:
            self.t_unfav += self.dt_hours
            if self.t_unfav > 24:
                self.rh_max = 0.0

            if self.t_unfav <= 6:
                dm_dec = -0.00133 * p["cmat"] * self.dt_hours
            elif self.t_unfav <= 24:
                dm_dec = 0.0
            else:
                dm_dec = -0.000667 * p["cmat"] * self.dt_hours

            return max(0.0, m_prev + dm_dec)

    def _m_to_raw_level(self, m: float) -> AlertLevel:
        """Map mold index to raw alert level (no debounce)."""
        if m >= ENTRY_THRESHOLDS[AlertLevel.CRITICAL]:
            return AlertLevel.CRITICAL
        if m >= ENTRY_THRESHOLDS[AlertLevel.WARNING]:
            return AlertLevel.WARNING
        if m >= ENTRY_THRESHOLDS[AlertLevel.MONITOR]:
            return AlertLevel.MONITOR
        return AlertLevel.SAFE

    def _resolve_level(self, target: AlertLevel) -> AlertLevel:
        """
        Apply debounce (entry) and hysteresis (exit) logic.

        Entry: level only activates after DEBOUNCE_HOURS consecutive triggers.
        Exit:  level only clears when M drops below EXIT_THRESHOLDS.
        """
        resolved = self.current_level

        # Check escalation (going up)
        for level in [AlertLevel.CRITICAL,
                      AlertLevel.WARNING,
                      AlertLevel.MONITOR]:
            if target.value >= level.value:
                self.debounce_counters[level] += self.dt_hours
                if self.debounce_counters[level] >= DEBOUNCE_HOURS[level]:
                    resolved = max(resolved, level,
                                   key=lambda x: x.value)
            else:
                self.debounce_counters[level] = 0

        # Check de-escalation (going down via hysteresis)
        if resolved != AlertLevel.SAFE:
            exit_thresh = EXIT_THRESHOLDS.get(resolved, 0.0)
            if self.M < exit_thresh:
                # Step down one level at a time
                resolved = AlertLevel(max(0, resolved.value - 1))

        return resolved

    @staticmethod
    def _message(level: AlertLevel) -> str:
        return {
            AlertLevel.SAFE:     "No mold risk detected.",
            AlertLevel.MONITOR:  "Microscopic growth detected. Monitor closely.",
            AlertLevel.WARNING:  "Mold approaching visible threshold. "
                                 "Increase ventilation or dehumidify.",
            AlertLevel.CRITICAL: "Heavy mold growth likely. "
                                 "Immediate inspection required.",
        }[level]


# ── CLI self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("AlertStateMachine — self test\n")

    asm = AlertStateMachine(material="sensitive", dt_hours=1.0)

    # Simulate 30 days of safe conditions
    print("Phase 1: Safe conditions (22°C cavity, 60% RH, 30 days)")
    for _ in range(30 * 24):
        s = asm.update(cavity_temp_c=22.0, cavity_rh_pct=60.0)
    print(f"  M={s['M']:.4f}  Level={s['level_name']}  "
          f"Message: {s['message']}")

    # Simulate 60 days of dangerous conditions
    print("\nPhase 2: Dangerous conditions (18°C cavity, 90% RH, 60 days)")
    first_monitor  = None
    first_warning  = None
    first_critical = None

    for hour in range(60 * 24):
        s = asm.update(cavity_temp_c=18.0, cavity_rh_pct=90.0)
        if first_monitor  is None and s['level'] == AlertLevel.MONITOR:
            first_monitor  = hour / 24
        if first_warning  is None and s['level'] == AlertLevel.WARNING:
            first_warning  = hour / 24
        if first_critical is None and s['level'] == AlertLevel.CRITICAL:
            first_critical = hour / 24

    print(f"  First MONITOR  : day {first_monitor:.1f}"
          if first_monitor  else "  MONITOR  : never triggered")
    print(f"  First WARNING  : day {first_warning:.1f}"
          if first_warning  else "  WARNING  : never triggered")
    print(f"  First CRITICAL : day {first_critical:.1f}"
          if first_critical else "  CRITICAL : never triggered")
    print(f"  Final M={s['M']:.4f}  Level={s['level_name']}")

    # Save results
    
    out_dir = Path(__file__).parent / "pyresults"
    out_dir.mkdir(exist_ok=True)
    
    results = {
        "test": "alert_state_self_test",
        "material": "sensitive",
        "phase1": {"M": 0.0, "level": "SAFE"},
        "phase2": {
            "first_monitor_day": first_monitor,
            "first_warning_day": first_warning,
            "first_critical_day": first_critical,
            "final_M": float(s['M']),
            "final_level": s['level_name'],
        },
        "phase3": {"M": float(s['M']), "level": s['level_name']},
        "summary": {k: (float(v) if hasattr(v, '__float__') else v)
                    for k, v in asm.summary.items()},
    }
    out_path = out_dir / "alert_state_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Simulate 30 days of recovery
    print("\nPhase 3: Recovery (22°C cavity, 55% RH, 30 days)")
    for _ in range(30 * 24):
        s = asm.update(cavity_temp_c=22.0, cavity_rh_pct=55.0)
    print(f"  M={s['M']:.4f}  Level={s['level_name']}  "
          f"Message: {s['message']}")

    print(f"\nSummary: {asm.summary}")