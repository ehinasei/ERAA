"""
fit_load_temperature_gam_2025.py
--------------------------------
Load-temperature relationship per region for 2025 WITHOUT assuming a cubic.

Instead of a fixed polynomial, each region gets a smooth curve fitted as a
penalised regression spline - the same kind of model as R's
mgcv::gam(L ~ s(T), method = "REML"):

  * daily average load is modelled as a smooth function of daily average
    temperature, built from 20 cubic B-spline pieces;
  * a roughness penalty stops the curve from wiggling;
  * how much smoothing to apply is chosen automatically by REML, so the data
    decide the shape. The "effective degrees of freedom" (EDF) tells you how
    complex the chosen curve is: EDF ~ 1 is a straight line and higher values
    mean more bends. EDF is NOT the same as a polynomial degree - a clearly
    curved U-shape like a cubic typically gets EDF ~ 4-7 - so judge the cubic
    by the cross-validation comparison, not by EDF alone.

A cubic polynomial is fitted alongside for comparison, and both are tested
out-of-sample (10-fold cross-validation, holding out whole weeks).

Outside the 2025 temperature range the curves are held flat at their edge
values (no extrapolation), ready for the climate-year step.

Inputs
  LOAD_FILE   Historical hourly electricity load.xlsx  (sheets 'Right bank', 'Left bank')
  TEMP_FILE   temp_output/temperature_by_region_2025.xlsx

Outputs (OUT_DIR)
  load_temperature_gam_2025.xlsx
      Model summary      EDF, R2, CV error GAM vs cubic, smoothing parameter...
      GAM curve (MW)     load at every 0.5 degC from -25 to 42 (flat outside range)
      GAM curve (rel)    same divided by the region's mean daily load ->
                         use for normalisation: f(T_actual) / f(T_normal)
      GAM 95% band       lower / upper confidence band (MW)
      Cubic curve (MW)   cubic for comparison (also flat outside range)
      Daily data / Cleaning log / Notes
  load_temperature_gam_2025.png   scatter + GAM (with band) + cubic per region


"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- settings
LOAD_FILE = "Historical hourly electricity load.xlsx"
TEMP_FILE = os.path.join("temp_output", "temperature_by_region_2025.xlsx")
OUT_DIR = "fit_output"
YEAR = 2025
LEFT_BANK_NAME = "Dnestroenergo"     # region name of the left bank in the GeoJSON

N_SEGMENTS = 20                      # B-spline pieces (more than needed; penalty does the rest)
T_GRID = np.round(np.arange(-25, 42.01, 0.5), 1)   # output temperatures (deg C)
CV_FOLDS = 10
SEED = 1

FIXES = [   # known single-hour errors (SEI data notes)
    ("C", "2025-02-04 18:00", "interp", "spike ~18x neighbours"),
    ("C", "2025-11-20 15:00", "interp", "single-hour dip"),
    ("C", "2025-12-12 03:00", "interp", "single-hour dip"),
    (LEFT_BANK_NAME, "2025-10-26 03:00", "halve", "DST fall-back: two hours summed"),
]
# --------------------------------------------------------------------------


# ================================================================ data
def read_load():
    rb = pd.read_excel(LOAD_FILE, sheet_name="Right bank", header=None)
    lb = pd.read_excel(LOAD_FILE, sheet_name="Left bank", header=None)

    col0 = next(c for c in rb.columns if str(rb.iat[0, c]).split(".")[0] == str(YEAR))
    block = rb.iloc[2:, col0:col0 + 7].dropna(how="all")
    names = [str(n).replace("Region", "").strip() for n in rb.iloc[1, col0 + 1:col0 + 7]]
    t = pd.to_datetime(block.iloc[:, 0]).dt.floor("h")
    load = pd.DataFrame(block.iloc[:, 1:].astype(float).values, index=t, columns=names)

    lcol = next(c for c in lb.columns if str(lb.iat[1, c]).split(".")[0] == str(YEAR))
    lblock = lb.iloc[2:, lcol:lcol + 2].dropna(how="all")
    lt = pd.to_datetime(lblock.iloc[:, 0]).dt.floor("h")
    load[LEFT_BANK_NAME] = pd.Series(lblock.iloc[:, 1].astype(float).values, index=lt)
    return load


def clean(load):
    log = []
    for reg, ts, action, why in FIXES:
        ts = pd.Timestamp(ts)
        if ts not in load.index or reg not in load.columns:
            continue
        old = load.at[ts, reg]
        load.at[ts, reg] = old / 2 if action == "halve" else np.nan
        log.append((reg, ts, old, action, why))
    load = load.interpolate(limit=3)
    log = [dict(region=r, hour=ts, original_MW=round(o, 1), new_MW=round(load.at[ts, r], 1),
                action=a, reason=w) for r, ts, o, a, w in log]
    return load, pd.DataFrame(log)


def read_temperature():
    t = pd.read_excel(TEMP_FILE)
    t.index = pd.to_datetime(t["time_local_UTC+2"])
    t = t.drop(columns=["time_utc", "time_local_UTC+2"])
    t.columns = [c.replace(" (degC)", "") for c in t.columns]
    return t


# ================================================================ smooth (GAM) model
def bspline_basis(x, lo, hi, nseg=N_SEGMENTS, deg=3):
    """Cubic B-spline basis on equally spaced knots (Cox-de Boor recursion)."""
    x = np.asarray(x, dtype=float)
    dx = (hi - lo) / nseg
    knots = lo + dx * np.arange(-deg, nseg + deg + 1)
    B = ((x[:, None] >= knots[None, :-1]) & (x[:, None] < knots[None, 1:])).astype(float)
    for k in range(1, deg + 1):
        left = (x[:, None] - knots[None, :-(k + 1)]) / (knots[None, k:-1] - knots[None, :-(k + 1)])
        right = (knots[None, k + 1:] - x[:, None]) / (knots[None, k + 1:] - knots[None, 1:-k])
        B = left * B[:, :-1] + right * B[:, 1:]
    return B


class SmoothFit:
    """Penalised regression spline, smoothing chosen by REML (as mgcv's s(T), method='REML')."""

    def __init__(self, T, L):
        T, L = np.asarray(T, float), np.asarray(L, float)
        pad = 1e-6 * (T.max() - T.min())
        self.lo, self.hi = T.min() - pad, T.max() + pad
        self.t_min, self.t_max = T.min(), T.max()
        B = bspline_basis(T, self.lo, self.hi)
        nb = B.shape[1]
        D = np.diff(np.eye(nb), 2, axis=0)         # 2nd-order difference penalty
        P = D.T @ D
        BtB, Bty = B.T @ B, B.T @ L
        n, r = len(L), nb - 2                       # penalty rank; null space = straight line
        nm = n - (nb - r)

        def solve(lam):
            A = BtB + lam * P
            beta = np.linalg.solve(A, Bty)
            res = L - B @ beta
            rss, pen = res @ res, lam * beta @ P @ beta
            sigma2 = (rss + pen) / nm
            reml = nm * np.log(sigma2) + np.linalg.slogdet(A)[1] - r * np.log(lam)
            return reml, beta, A, rss

        grid = 10.0 ** np.linspace(-4, 8, 121)
        scores = [solve(l)[0] for l in grid]
        i = int(np.argmin(scores))
        fine = 10.0 ** np.linspace(np.log10(grid[max(i - 1, 0)]),
                                   np.log10(grid[min(i + 1, len(grid) - 1)]), 81)
        scores_f = [solve(l)[0] for l in fine]
        self.lam = fine[int(np.argmin(scores_f))]
        self.lam_at_bound = i in (0, len(grid) - 1)

        _, self.beta, A, rss = solve(self.lam)
        Ainv = np.linalg.inv(A)
        self.edf = float(np.trace(Ainv @ BtB))     # includes intercept
        self.sigma2 = rss / (n - self.edf)
        self.Vb = self.sigma2 * Ainv               # Bayesian covariance, as in mgcv
        self.r2 = 1 - rss / np.sum((L - L.mean()) ** 2)
        self.r2_adj = 1 - (rss / (n - self.edf)) / (np.sum((L - L.mean()) ** 2) / (n - 1))
        self.aic = n * np.log(rss / n) + 2 * self.edf
        self.n = n

    def predict(self, T, se=False):
        Tc = np.clip(np.asarray(T, float), self.t_min, self.t_max)   # hold flat outside range
        Bg = bspline_basis(Tc, self.lo, self.hi)
        y = Bg @ self.beta
        if not se:
            return y
        return y, np.sqrt(np.einsum("ij,jk,ik->i", Bg, self.Vb, Bg))


class CubicFit:
    def __init__(self, T, L):
        T, L = np.asarray(T, float), np.asarray(L, float)
        self.c = np.polyfit(T, L, 3)
        self.t_min, self.t_max = T.min(), T.max()
        rss = np.sum((L - np.polyval(self.c, T)) ** 2)
        n = len(L)
        self.r2 = 1 - rss / np.sum((L - L.mean()) ** 2)
        self.aic = n * np.log(rss / n) + 2 * 4

    def predict(self, T):
        return np.polyval(self.c, np.clip(np.asarray(T, float), self.t_min, self.t_max))


def cv_rmse(model_cls, T, L, folds):
    err = []
    for f in np.unique(folds):
        tr = folds != f
        m = model_cls(T[tr], L[tr])
        err.append(L[~tr] - m.predict(T[~tr]))
    return float(np.sqrt(np.mean(np.concatenate(err) ** 2)))


def week_folds(days):
    rng = np.random.default_rng(SEED)
    wk = pd.Index(days).isocalendar()
    key = (wk["year"].astype(str) + "-" + wk["week"].astype(str)).values
    uniq = pd.unique(key)
    fold = dict(zip(uniq, rng.permutation(len(uniq)) % CV_FOLDS))
    return np.array([fold[k] for k in key])


# ================================================================ output helpers
def save_safely(write_fn, path):
    try:
        write_fn(path)
        return path
    except (PermissionError, OSError) as e:
        stem, ext = os.path.splitext(path)
        alt = f"{stem}_{pd.Timestamp.now():%H%M%S}{ext}"
        print(f"WARNING: could not write {path} ({e}). Is it open? Saving as {alt}")
        write_fn(alt)
        return alt


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    load, log = clean(read_load())
    temp = read_temperature()

    regions = [r for r in load.columns if r in temp.columns]
    load["NATIONAL"] = load[regions].sum(axis=1, min_count=len(regions))
    regions_all = regions + ["NATIONAL"]

    dl = load[regions_all].resample("D").mean()
    dt = temp[regions_all].resample("D").mean()
    days = dl.index.intersection(dt.index)
    dl, dt = dl.loc[days], dt.loc[days]
    folds = week_folds(days)

    summary, gam_mw, gam_rel, band_lo, band_hi, cub_mw, fits = [], {}, {}, {}, {}, {}, {}
    for r in regions_all:
        ok = (dl[r].notna() & dt[r].notna()).values
        T, L = dt[r].values[ok], dl[r].values[ok]
        g, c = SmoothFit(T, L), CubicFit(T, L)
        fits[r] = (g, c, T, L)
        cv_g = cv_rmse(SmoothFit, T, L, folds[ok])
        cv_c = cv_rmse(CubicFit, T, L, folds[ok])

        y, se = g.predict(T_GRID, se=True)
        inside = (T_GRID >= T.min()) & (T_GRID <= T.max())
        gam_mw[r], gam_rel[r] = y, y / L.mean()
        band_lo[r], band_hi[r] = y - 1.96 * se, y + 1.96 * se
        cub_mw[r] = c.predict(T_GRID)

        fine = np.linspace(T.min(), T.max(), 1001)
        yf = g.predict(fine)
        summary.append({
            "region": r, "n_days": int(ok.sum()),
            "EDF_smooth": round(g.edf - 1, 2),          # mgcv reports EDF without intercept
            "R2_GAM": g.r2, "R2adj_GAM": g.r2_adj, "R2_cubic": c.r2,
            "CV_RMSE_GAM_MW": cv_g, "CV_RMSE_cubic_MW": cv_c,
            "CV_cubic_vs_GAM_%": 100 * (cv_c / cv_g - 1),
            "AIC_GAM": g.aic, "AIC_cubic": c.aic,
            "better_out_of_sample": "GAM" if cv_g < cv_c else "cubic",
            "mean_daily_load_MW": L.mean(),
            "T_min_degC": T.min(), "T_max_degC": T.max(),
            "T_at_min_load_degC": fine[np.argmin(yf)],
            "load_at_T_min_%_of_mean": 100 * yf[0] / L.mean(),
            "load_at_T_max_%_of_mean": 100 * yf[-1] / L.mean(),
            "smoothing_lambda": g.lam,
            "warning": "smoothing at search limit" if g.lam_at_bound else "",
        })

    summ = pd.DataFrame(summary)
    print(summ[["region", "EDF_smooth", "R2_GAM", "R2_cubic", "CV_cubic_vs_GAM_%",
                "better_out_of_sample", "T_at_min_load_degC"]].round(3).to_string(index=False))

    # ---- Excel
    def grid_df(d):
        df = pd.DataFrame(d, index=pd.Index(T_GRID, name="T_degC"))
        return df

    obs = pd.DataFrame({r: ["" if a <= tt <= b else "held flat" for tt in T_GRID]
                        for r, a, b in zip(summ.region, summ.T_min_degC, summ.T_max_degC)},
                       index=pd.Index(T_GRID, name="T_degC"))
    band = pd.concat({"lower95": grid_df(band_lo), "upper95": grid_df(band_hi)}, axis=1)
    band.columns = [f"{r} {k}" for k, r in band.columns]
    daily = pd.concat({"load_MW": dl, "temp_degC": dt}, axis=1)
    daily.columns = [f"{r} {k}" for k, r in daily.columns]
    daily.insert(0, "weekday", daily.index.day_name())
    daily.index.name = "date"
    notes = pd.DataFrame({"note": [
        f"Smooth (GAM) fit of daily average load vs daily average temperature per region, {YEAR}.",
        "Penalised cubic regression spline (20 B-spline segments, 2nd-order difference penalty); "
        "smoothing chosen by REML - equivalent in spirit to mgcv::gam(L ~ s(T), method='REML').",
        "EDF_smooth = effective degrees of freedom of the curve (excl. intercept): ~1 = straight line, "
        "higher = more bends. Not equal to a polynomial degree (a cubic-shaped U often gives EDF 4-7); "
        "compare models with the CV columns.",
        "There is no formula for a GAM curve: use the 'GAM curve' tables as look-up tables "
        "(interpolate between 0.5 degC steps).",
        "Normalisation (USAID step 2/4): factor = f_rel(T_actual) / f_rel(T_normal), using 'GAM curve (rel)'.",
        "Curves are held flat outside the 2025 observed temperature range (column flags in 'Observed range').",
        "CV = 10-fold cross-validation, whole weeks held out; lower CV_RMSE = better predictions on unseen days.",
        "Load in A, B, D, E, F is NET of distributed PV; Dnestroenergo 2025 shows no winter peak "
        "(probably the 2025 left-bank energy crisis) - treat its curve with caution.",
        "Both load sheets are read as fixed UTC+2, matching the temperature file.",
    ]})

    def write_xlsx(path):
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            summ.round(4).to_excel(xw, sheet_name="Model summary", index=False)
            grid_df(gam_mw).round(2).to_excel(xw, sheet_name="GAM curve (MW)")
            grid_df(gam_rel).round(5).to_excel(xw, sheet_name="GAM curve (rel)")
            band.round(2).to_excel(xw, sheet_name="GAM 95% band")
            grid_df(cub_mw).round(2).to_excel(xw, sheet_name="Cubic curve (MW)")
            obs.to_excel(xw, sheet_name="Observed range")
            daily.round(2).to_excel(xw, sheet_name="Daily data")
            log.to_excel(xw, sheet_name="Cleaning log", index=False)
            notes.to_excel(xw, sheet_name="Notes", index=False)
            for ws in xw.sheets.values():
                ws.freeze_panes = "B2"
                for col in ws.columns:
                    ws.column_dimensions[col[0].column_letter].width = \
                        max(11, min(70, len(str(col[0].value)) + 2))

    print("written", save_safely(write_xlsx, os.path.join(OUT_DIR, f"load_temperature_gam_{YEAR}.xlsx")))

    # ---- chart
    n, ncol = len(regions_all), 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 4.3 * nrow), constrained_layout=True)
    pt, gam_c, cub_c, muted = "#2a78d6", "#0b0b0b", "#eb6834", "#52514e"
    for ax, r in zip(axes.ravel(), regions_all):
        g, c, T, L = fits[r]
        s = summ.set_index("region").loc[r]
        xs = np.linspace(T.min(), T.max(), 300)
        y, se = g.predict(xs, se=True)
        ax.scatter(T, L, s=12, color=pt, alpha=0.45, linewidths=0)
        ax.fill_between(xs, y - 1.96 * se, y + 1.96 * se, color=gam_c, alpha=0.12, linewidth=0)
        ax.plot(xs, y, color=gam_c, lw=2, label="GAM (smooth, REML)")
        ax.plot(xs, c.predict(xs), color=cub_c, lw=2, ls="--", label="Cubic")
        ax.set_title("National (sum of regions)" if r == "NATIONAL" else r,
                     loc="left", fontsize=12, fontweight="bold")
        ax.text(0.98, 0.96, f"EDF = {s.EDF_smooth:.1f}\nR² GAM {s.R2_GAM:.2f} | cubic {s.R2_cubic:.2f}"
                f"\nbetter out-of-sample: {s.better_out_of_sample}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=muted)
        ax.set_xlabel("Daily average temperature (°C)", fontsize=9, color=muted)
        ax.set_ylabel("Daily average load (MW)", fontsize=9, color=muted)
        ax.tick_params(colors=muted, labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#c9c8c2")
        ax.grid(axis="y", color="#ecebe7", lw=0.8)
        ax.set_axisbelow(True)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=10)
    fig.suptitle(f"Daily load vs temperature, {YEAR}: smooth GAM fit (shaded = 95% band) vs cubic",
                 x=0.01, ha="left", fontsize=13)

    def write_png(path):
        fig.savefig(path, dpi=150)
    print("written", save_safely(write_png, os.path.join(OUT_DIR, f"load_temperature_gam_{YEAR}.png")))
    plt.close(fig)


if __name__ == "__main__":
    main()
