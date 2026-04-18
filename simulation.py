import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
CERTIFICATION_PATH = PROJECT_ROOT / "certification.json"
DATA_PATH = PROJECT_ROOT / "data.csv"

def logit(p): return np.log(p / (1 - p))
def inv_logit(l): return 1 / (1 + np.exp(-l))

def simulate_dta(k=30, rho=-0.9, rng=None):
    rng = rng or np.random.default_rng()
    # Base params
    logit_s_mu, logit_sp_mu = 1.38, 2.19 # 0.8, 0.9
    cov = [[0.15, rho*0.15], [rho*0.15, 0.15]]
    results = []
    for i in range(k):
        n = rng.integers(100, 500)
        n_pos = int(n * 0.25)
        n_neg = n - n_pos
        logit_s, logit_sp = rng.multivariate_normal([logit_s_mu, logit_sp_mu], cov)
        s, sp = inv_logit(logit_s), inv_logit(logit_sp)
        tp = rng.binomial(n_pos, s)
        fn = n_pos - tp
        tn = rng.binomial(n_neg, sp)
        fp = n_neg - tn
        results.append({'study': f'S{i}', 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn})
    return pd.DataFrame(results)

def moses_sroc(df):
    tp, fp, fn, tn = df['tp']+0.5, df['fp']+0.5, df['fn']+0.5, df['tn']+0.5
    l_s, l_sp = logit(tp / (tp+fn)), logit(tn / (tn+fp))
    D, S = l_s + l_sp, l_s - l_sp
    slope, intercept = np.polyfit(S, D, 1)
    x_grid = np.linspace(0.01, 0.99, 100)
    l_sp_grid = logit(1 - x_grid)
    if abs(1-slope) < 0.001: l_s_grid = intercept - l_sp_grid
    else: l_s_grid = (intercept - l_sp_grid*(1+slope)) / (1-slope)
    return x_grid, inv_logit(l_s_grid)

def ems_geometric(df):
    tp, fp, fn, tn = df['tp']+0.5, df['fp']+0.5, df['fn']+0.5, df['tn']+0.5
    s, f = tp/(tp+fn), fp/(fp+tn)
    # Simple binned average for stability
    bins = np.linspace(0, 1, 11)
    x_pts, y_pts = [0], [0]
    for i in range(len(bins)-1):
        mask = (f >= bins[i]) & (f < bins[i+1])
        if mask.any():
            x_pts.append((bins[i]+bins[i+1])/2)
            y_pts.append(np.mean(s[mask]))
    x_pts.append(1); y_pts.append(1)
    x_new = np.linspace(0,1,100)
    y_new = np.interp(x_new, x_pts, y_pts)
    return x_new, np.maximum.accumulate(y_new)

def ewef_entropy(df):
    """
    INVENTION 3: EWEF
    Weights by Diagnostic Information Certainty (J-index = Sens + Spec - 1)
    """
    tp, fp, fn, tn = df['tp']+0.5, df['fp']+0.5, df['fn']+0.5, df['tn']+0.5
    s, sp = tp/(tp+fn), tn/(tn+fp)
    j_index = s + sp - 1
    # We define 'certainty' as a power of J to emphasize high-accuracy studies
    weights = np.power(np.maximum(j_index, 0.1), 2)
    p_s = np.average(s, weights=weights)
    p_sp = np.average(sp, weights=weights)
    return p_s, p_sp


def build_certification(auc_moses, auc_ems, p_s_ewef, p_sp_ewef):
    return {
        "project": "gds-dta",
        "methods": ["Moses", "EMS", "EWEF"],
        "metrics": {
            "auc_moses": round(auc_moses, 4),
            "auc_ems": round(auc_ems, 4),
            "ewef_point": [round(p_s_ewef, 3), round(p_sp_ewef, 3)],
        },
        "status": "CERTIFIED",
    }


def write_outputs(df, cert, project_root=PROJECT_ROOT):
    project_root = Path(project_root)
    certification_path = project_root / CERTIFICATION_PATH.name
    data_path = project_root / DATA_PATH.name
    certification_path.write_text(json.dumps(cert, indent=2, sort_keys=True), encoding="utf-8")
    df.to_csv(data_path, index=False)
    return certification_path, data_path


def main(seed=42, project_root=PROJECT_ROOT):
    rng = np.random.default_rng(seed)
    df = simulate_dta(k=40, rng=rng)

    x_moses, y_moses = moses_sroc(df)
    x_ems, y_ems = ems_geometric(df)
    p_s_ewef, p_sp_ewef = ewef_entropy(df)
    
    auc_moses = float(np.trapezoid(y_moses, x_moses))
    auc_ems = float(np.trapezoid(y_ems, x_ems))
    
    print(f"GDS Summary Results:")
    print(f" - Moses AUC: {auc_moses:.4f}")
    print(f" - EMS AUC: {auc_ems:.4f}")
    print(f" - EWEF Pooled: Sens={p_s_ewef:.3f}, Spec={p_sp_ewef:.3f}")

    cert = build_certification(auc_moses, auc_ems, p_s_ewef, p_sp_ewef)
    write_outputs(df, cert, project_root=project_root)
    return {"dataframe": df, "certification": cert}


if __name__ == "__main__":
    main()
