import numpy as np


def rh_crit(temp_c):
    temp_c = np.asarray(temp_c, dtype=float)
    return np.where(temp_c < 0, 1.0,
           np.where(temp_c < 5, 0.95,
           np.where(temp_c < 10, 0.90,
           np.where(temp_c < 15, 0.85,
           np.where(temp_c < 20, 0.80,
           np.where(temp_c < 25, 0.75, 0.70))))))


def e_sat(t):
    return 6.1078 * np.exp(17.27 * t / (t + 237.3))


def surface_rh(surface_temp_c, outdoor_temp_c, outdoor_rh_pct):
    e_out = (outdoor_rh_pct / 100.0) * e_sat(outdoor_temp_c)
    return np.clip(e_out / e_sat(surface_temp_c), 0.0, 1.0)


def compute_moisture_index(surface_temp_c, outdoor_temp_c,
                           outdoor_rh_pct, dt_hours=1/60,
                           material="drywall"):
    sensitivity = {"drywall": 1.0, "wood": 0.8, "concrete": 0.5}
    k = sensitivity.get(material, 1.0)
    surface_temp_c = np.asarray(surface_temp_c, dtype=float)
    outdoor_temp_c = np.asarray(outdoor_temp_c, dtype=float)
    outdoor_rh_pct = np.asarray(outdoor_rh_pct, dtype=float)
    n = len(surface_temp_c)
    rh_surf = surface_rh(surface_temp_c, outdoor_temp_c, outdoor_rh_pct)
    rh_cr   = rh_crit(surface_temp_c)
    excess  = rh_surf - rh_cr
    mi = np.zeros(n)
    for i in range(1, n):
        if excess[i] > 0:
            mi[i] = mi[i-1] + k * excess[i] * dt_hours
        else:
            mi[i] = max(0.0, mi[i-1] + k * 0.1 * excess[i] * dt_hours)
    return {
        "rh_surface": rh_surf, "rh_crit": rh_cr,
        "exceedance": excess,  "mi": mi,
        "risk_score": np.clip(mi / 720.0 * 100, 0, 100),
        "alert":      mi > 24.0,
    }
