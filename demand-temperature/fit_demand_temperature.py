"""
fit_demand_temperature.py
-------------------------
Demand-temperature relationship per region, 2021-2025 (USAID 2024 method,
step 1: report p.34, Figure 30), as ONE CUBIC POLYNOMIAL PER REGION PER YEAR,
then averaged across the years.

1. Daily values. For each region and each calendar day (fixed UTC+2):
       daily load        = mean of the day's 24 hourly loads (MW)
       daily temperature = mean of the day's 24 hourly regional temperatures (deg C;
                           region_mean_K from polynomial_fit.py = mean of the region's cells)
   -> about 365 (temperature, load) points per region per year.

2. Yearly cubic. Least-squares fit for every region and year:
       load(T) = a3*T^3 + a2*T^2 + a1*T + a0                       (MW)
   Relative form: every coefficient divided by that year's mean daily load,
       load(T) / mean load = a3_rel*T^3 + a2_rel*T^2 + a1_rel*T + a0_rel
   which removes differences in load LEVEL between years (growth, crisis years)
   and keeps only the temperature response.

3. Average cubic per region = mean of the yearly RELATIVE coefficients over the
   years used. (Averaging the coefficients of same-degree polynomials gives exactly
   the point-by-point average of the yearly curves.) The mean of the absolute (MW)
   coefficients is also given, for reference.
   Valid range = the temperatures covered by EVERY year used (highest yearly minimum
   to lowest yearly maximum). Outside it, hold the curve flat at the edge value.


Time alignment - everything on fixed UTC+2 (no daylight saving):
    * Right bank: already fixed UTC+2. Timestamps hh:30 -> hour hh.
    * Left bank: local clock WITH daylight saving (Europe/Chisinau). Summer hours are
      moved one hour earlier; the non-existent spring hour (a filler copy in the file)
      is dropped; the autumn hour that occurs twice in local time is restored - split in
      half if the two hours were summed into one row (2024, 2025), else interpolated.
    * Temperature: temperature_by_region_<year>.xlsx from polynomial_fit.py (UTC+2).

Inputs (same folder as this script)
    Historical hourly electricity load.xlsx
    temperature_values/temperature_by_region_<year>.xlsx   (one tab per region)

Outputs (fit_output/)
    demand_temperature_cubic_2021_2025.xlsx
        Average cubic        per region: averaged relative coefficients, valid range,
                             years used (+ averaged MW coefficients for reference)
        Cubic by year        per region x year: coefficients (MW and relative), R2,
                             temperature range, status (used / excluded and why)
        Average curve (rel)  the averaged cubic every 0.5 degC, held flat outside the
                             valid range - ready for normalisation: f(T_actual)/f(T_normal)
        Daily data / Data fixes / Notes
    demand_temperature_cubic_2021_2025.png   per region: yearly cubics + average (relative)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- settings
BASE = Path(__file__).resolve().parent                    # demand-temperature
LOAD_FILE = BASE / "Historical hourly electricity load.xlsx"
TEMP_DIR = BASE / "temperature_values"
OUT_DIR = BASE / "fit_output"
YEARS = range(2021, 2026)
FIXED_TZ = "Etc/GMT-2"            # fixed UTC+2
LEFT_BANK_TZ = "Europe/Chisinau"  # local time with DST

DEGREE = 3                        # polynomial degree used for every region and year
RUN_MODEL_COMPARISON = True       # also test polynomials of degree 1-5 (diagnostic only)
MODELS = ["poly1", "poly2", "poly3", "poly4", "poly5"]
CV_FOLDS = 10
SEED = 1
T_GRID = np.round(np.arange(-25, 42.01, 0.5), 1)   # temperatures for the curve table (deg C)

LEFT_BANK_TAB = None   # name of the left-bank tab in the temperature workbooks;
                       # None = the one tab that isn't A-F
INCLUDE_NATIONAL = True

# Region-years left out of the fits (listed with their reason in "Cubic by year")
EXCLUDE = {
    (2024, "A"): "2024 is a scaled copy of 2023 (r = 0.999)",
    (2024, "B"): "2024 is an exact copy of 2023",
    (2024, "D"): "2024 is an exact copy of 2023",
    (2024, "E"): "2024 is an exact copy of 2023",
    (2024, "F"): "2024 is an exact copy of 2023",
    (2024, "NATIONAL"): "includes copied 2024 data (A, B, D, E, F)",
}
# Days left out of the fits (events / filled data)
EXCLUDE_DAYS = {
    "2022-11-15": "grid emergency (strikes on Ukraine's grid)",
    "2022-11-23": "blackout",
    "2022-10-30": "left bank: whole day copied from 2023-10-29",
}
# Single-hour errors (SEI data notes), given in the ORIGINAL timestamps of the file
SPIKES = [
    ("C", "2025-02-04 18:30"), ("C", "2025-11-20 15:30"), ("C", "2025-12-12 03:30"),
    ("LEFT", "2023-09-09 02:30"),
]
# --------------------------------------------------------------------------


# ================================================================ load data
def to_timestamps(col):
    """Excel dates"""
    num = pd.to_numeric(col, errors="coerce")
    if num.notna().mean() > 0.9:
        return pd.to_datetime(num, unit="D", origin="1899-12-30").dt.round("min")
    return pd.to_datetime(col, errors="coerce")


def year_of(cell):
    try:
        return int(float(cell))
    except (TypeError, ValueError):
        return None


def read_right_bank(year, raw, fixes):
    col0 = next((c for c in raw.columns if year_of(raw.iat[0, c]) == year), None)
    if col0 is None:
        return None
    block = raw.iloc[2:, col0:col0 + 7]
    t = to_timestamps(block.iloc[:, 0])
    ok = t.notna()
    names = [str(n).replace("Region", "").strip() for n in raw.iloc[1, col0 + 1:col0 + 7]]
    df = pd.DataFrame(block.loc[ok].iloc[:, 1:].astype(float).values, index=t[ok].values, columns=names)
    for reg, ts in SPIKES:
        ts = pd.Timestamp(ts)
        if reg in df.columns and ts in df.index:
            fixes.append(dict(series=reg, time=ts, value_MW=round(df.at[ts, reg], 1),
                              action="single-hour error -> interpolated"))
            df.at[ts, reg] = np.nan
    df.index = df.index.floor("h")                      # fixed UTC+2, hh:30 -> hour hh
    return df.interpolate(limit=3)


def read_left_bank(year, raw, fixes):
    col = next((c for c in raw.columns if year_of(raw.iat[1, c]) == year), None)
    if col is None:
        return None
    t = to_timestamps(raw.iloc[2:, col])
    v = pd.to_numeric(raw.iloc[2:, col + 1], errors="coerce")
    ok = t.notna()
    s = pd.Series(v[ok].values, index=t[ok].values, dtype=float)
    for reg, ts in SPIKES:
        ts = pd.Timestamp(ts)
        if reg == "LEFT" and ts in s.index:
            fixes.append(dict(series="Left bank", time=ts, value_MW=round(s[ts], 1),
                              action="single-hour error -> interpolated"))
            s[ts] = np.nan

    # local clock with DST -> fixed UTC+2
    label = s.index.floor("h")
    local = label.tz_localize(LEFT_BANK_TZ, ambiguous=np.ones(len(label), bool),  # repeated hour = summer time
                              nonexistent="NaT")
    dropped = label[local.isna()]
    for ts in dropped:
        fixes.append(dict(series="Left bank", time=ts, value_MW=round(s[label == ts].iloc[0], 1),
                          action="spring DST: non-existent local hour (filler copy) dropped"))
    keep = ~local.isna()
    s = pd.Series(s.values[keep], index=local[keep].tz_convert(FIXED_TZ).tz_localize(None))
    s = s[~s.index.duplicated()]

    # autumn: the hour lost by the shift -> split a summed (doubled) neighbour, or interpolate
    full = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    s = s.reindex(full)
    h = pd.Timedelta("1h")
    for gap in s.index[s.isna()]:
        if gap.month != 10:
            continue
        # the row next to the gap may hold the two local hours summed together
        split = False
        for nb, other in ((gap - h, gap - 2 * h), (gap + h, gap + 2 * h)):
            if nb in s.index and other in s.index and \
                    s[nb] > 1.6 * np.nanmean([s[other], s.get(gap + (gap - nb), np.nan)]):
                fixes.append(dict(series="Left bank", time=nb, value_MW=round(s[nb], 1),
                                  action="autumn DST: two hours summed in one row -> split in half"))
                s[gap] = s[nb] = s[nb] / 2
                split = True
                break
        if not split:
            fixes.append(dict(series="Left bank", time=gap, value_MW=np.nan,
                              action="autumn DST: missing hour after shift -> interpolated"))
    return s.interpolate(limit=3)


def read_load():
    rb = pd.read_excel(LOAD_FILE, sheet_name="Right bank", header=None)
    lb = pd.read_excel(LOAD_FILE, sheet_name="Left bank", header=None)
    fixes, out = [], {}
    for y in YEARS:
        r = read_right_bank(y, rb, fixes)
        l = read_left_bank(y, lb, fixes)
        if r is None or l is None:
            print(f"{y}: load data not found - skipped")
            continue
        r = r[~r.index.duplicated()]
        r["LEFT"] = l
        out[y] = r
    return out, pd.DataFrame(fixes)


# ================================================================ temperature
def read_temperature(year):
    path = TEMP_DIR / f"temperature_by_region_{year}.xlsx"
    if not path.exists():
        print(f"{year}: {path.name} not found - skipped")
        return None
    tabs = pd.read_excel(path, sheet_name=None, index_col=0)
    temps, all_cells = {}, []
    for reg, df in tabs.items():
        df.index = pd.to_datetime(df.index)
        temps[reg] = df["region_mean_K"] - 273.15                     # K -> deg C
        cells = df.drop(columns=[c for c in ("time_utc", "region_mean_K") if c in df.columns])
        all_cells.append(cells)
    t = pd.DataFrame(temps)
    if INCLUDE_NATIONAL:
        t["NATIONAL"] = pd.concat(all_cells, axis=1).mean(axis=1) - 273.15   # all cells of Moldova
    return t


def match_regions(load_cols, temp_cols):
    """Load column name -> temperature tab name."""
    m = {c: c for c in load_cols if c in temp_cols}
    left = LEFT_BANK_TAB or next(iter(set(temp_cols) - set(m.values()) - {"NATIONAL"}), None)
    if left is None:
        sys.exit("ERROR: can't tell which temperature tab is the left bank - set LEFT_BANK_TAB.")
    m["LEFT"] = left
    return m


# ================================================================ polynomial model
class Poly:
    def __init__(self, deg):
        self.deg = deg

    def fit(self, T, L):
        self.lo, self.hi = T.min(), T.max()
        self.c = np.polyfit(T, L, self.deg)
        self.k = self.deg + 1
        return self

    def predict(self, T):
        return np.polyval(self.c, np.clip(T, self.lo, self.hi))

    def describe(self):
        return "coefficients (highest power first): " + ", ".join(f"{v:.6g}" for v in self.c)


def make(name):
    return Poly(int(name[4:]))


def week_folds(index):
    rng = np.random.default_rng(SEED)
    wk = pd.DatetimeIndex(index).isocalendar()
    key = (wk["year"].astype(str) + "-" + wk["week"].astype(str)).values
    uniq = pd.unique(key)
    fold = dict(zip(uniq, rng.permutation(len(uniq)) % CV_FOLDS))
    return np.array([fold[k] for k in key])


def plausibility(model, T, L, tol=0.015):
    """Check the curve is U-shaped (or one-sided): going colder from its lowest point
    load should keep rising, and going hotter from its lowest point load should keep
    rising. Dips larger than `tol` x mean load (default 1.5%) are flagged."""
    xs = np.linspace(T.min(), T.max(), 400)
    y = model.predict(xs)
    i = int(np.argmin(y))
    span = tol * np.mean(np.abs(L))
    flags = []
    cold = y[:i + 1][::-1]                      # from the minimum towards the coldest day
    hot = y[i:]                                 # from the minimum towards the hottest day
    if len(cold) > 1 and np.max(np.maximum.accumulate(cold) - cold) > span:
        flags.append("bends back on the cold side (load falls as it gets colder)")
    if len(hot) > 1 and np.max(np.maximum.accumulate(hot) - hot) > span:
        flags.append("bends back on the warm side (load falls as it gets hotter)")
    return "; ".join(flags)


def u_shape(y):
    """Shape correction for the curve tables: from the curve's lowest point, load is not
    allowed to fall as it gets colder or hotter - where the fitted curve bends back, it is
    held flat at its highest value instead (heating/cooling demand saturates)."""
    y = np.asarray(y, dtype=float).copy()
    i = int(np.argmin(y))
    y[:i + 1] = np.maximum.accumulate(y[:i + 1][::-1])[::-1]
    y[i:] = np.maximum.accumulate(y[i:])
    return y


def compare_models(T, L, index):
    """Fit every candidate, cross-validate, choose best and recommended."""
    folds = week_folds(index)
    n, sst = len(L), np.sum((L - L.mean()) ** 2)
    rows, fitted, fold_mse_all = [], {}, {}
    for name in MODELS:
        m = make(name).fit(T, L)
        fitted[name] = m
        sse = np.sum((L - m.predict(T)) ** 2)
        fold_mse, err = [], []
        for f in np.unique(folds):
            tr = folds != f
            mm = make(name).fit(T[tr], L[tr])
            e = L[~tr] - mm.predict(T[~tr])
            err.append(e)
            fold_mse.append(np.mean(e ** 2))
        fold_mse_all[name] = np.array(fold_mse)
        rows.append(dict(
            model=name, n_params=round(m.k, 2), R2=1 - sse / sst,
            adj_R2=1 - (sse / (n - m.k)) / (sst / (n - 1)),
            BIC=n * np.log(sse / n) + m.k * np.log(n),
            CV_RMSE=np.sqrt(np.mean(np.concatenate(err) ** 2)),
            flag=plausibility(m, T, L), parameters=m.describe()))
    comp = pd.DataFrame(rows)
    comp["CV_vs_best_%"] = 100 * (comp.CV_RMSE / comp.CV_RMSE.min() - 1)
    best = comp.loc[comp.CV_RMSE.idxmin()]
    # paired comparison with the best model on the same held-out weeks:
    # "as good as the best" = mean extra error no larger than one standard error of the difference
    within = []
    for name in comp.model:
        d = fold_mse_all[name] - fold_mse_all[best.model]
        se = np.std(d, ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        within.append(bool(d.mean() <= se))
    comp["within_1SE_of_best"] = within
    ok = comp[comp.within_1SE_of_best]
    rec = ok.sort_values(["n_params", "CV_RMSE"]).iloc[0]
    comp["best_by_CV"] = comp.model == best.model
    comp["recommended"] = comp.model == rec.model
    return comp, fitted, best.model, rec.model


# ================================================================ main
def main():
    OUT_DIR.mkdir(exist_ok=True)
    loads, fixes = read_load()
    year_rows, comp_rows, daily_all, regions = [], [], [], None

    for y, load in loads.items():
        temp = read_temperature(y)
        if temp is None:
            continue
        m = match_regions(list(load.columns), temp.columns)
        load = load.rename(columns=m)
        regs = list(m.values())
        if INCLUDE_NATIONAL:
            load["NATIONAL"] = load[regs].sum(axis=1, min_count=len(regs))
            regs = regs + ["NATIONAL"]
        regions = regions or regs

        # 1. daily values: mean of the 24 hours of each UTC+2 calendar day
        dl = load[regs].resample("D").mean()
        dt = temp[regs].resample("D").mean()
        days = dl.index.intersection(dt.index)
        bad_days = pd.to_datetime(list(EXCLUDE_DAYS))

        for r in regs:
            d = pd.DataFrame({"T_degC": dt.loc[days, r], "load_MW": dl.loc[days, r]}).dropna()
            reason = EXCLUDE.get((y, r), "")
            d["used_in_fit"] = (~d.index.isin(bad_days)) & (reason == "")
            daily_all.append(d.assign(region=r, year=y))
            if reason:
                year_rows.append(dict(region=r, year=y, status=f"excluded: {reason}"))
                print(f"{y} {r:>14}: excluded ({reason})")
                continue

            # 2. yearly cubic (MW) and its relative form (/ mean daily load)
            f = d[d.used_in_fit]
            T, L = f["T_degC"].values, f["load_MW"].values
            p = Poly(DEGREE).fit(T, L)
            c, mean_l = p.c, L.mean()
            r2 = 1 - np.sum((L - np.polyval(c, T)) ** 2) / np.sum((L - mean_l) ** 2)
            row = dict(region=r, year=y, status="used", n_days=len(f), R2=r2, mean_load_MW=mean_l,
                       T_min_degC=T.min(), T_max_degC=T.max(),
                       shape_flag=plausibility(p, T, L))
            for k, v in zip(range(DEGREE, -1, -1), c):
                row[f"a{k}"] = v
            for k, v in zip(range(DEGREE, -1, -1), c / mean_l):
                row[f"a{k}_rel"] = v
            if RUN_MODEL_COMPARISON:
                comp, _, best, rec = compare_models(T, L, f.index)
                comp_rows.append(comp.assign(region=r, year=y))
                row.update(best_degree_by_CV=best,
                           cubic_CV_vs_best_pct=comp.set_index("model").loc[f"poly{DEGREE}", "CV_vs_best_%"])
            year_rows.append(row)
            print(f"{y} {r:>14}: R2 {r2:.3f}" + (f"  (best degree by CV: {row['best_degree_by_CV']})"
                                                if RUN_MODEL_COMPARISON else ""))

    if not year_rows:
        sys.exit("Nothing to fit - check the temperature workbooks exist.")
    by_year = pd.DataFrame(year_rows)
    daily = pd.concat(daily_all)
    daily.index.name = "date"
    a_cols = [f"a{k}" for k in range(DEGREE, -1, -1)]
    rel_cols = [f"{a}_rel" for a in a_cols]

    # 3. average of the yearly cubics, per region
    avg_rows, curves = [], {}
    for r in regions:
        f = by_year[(by_year.region == r) & (by_year.status == "used")]
        if f.empty:
            continue
        t_lo, t_hi = f.T_min_degC.max(), f.T_max_degC.min()        # range covered by every year
        c_rel = f[rel_cols].mean().values
        c_abs = f[a_cols].mean().values
        # how well the averaged relative cubic describes all valid days (each day / its year's mean)
        d = daily[(daily.region == r) & daily.used_in_fit & daily.year.isin(f.year)]
        rel = d["load_MW"] / d["year"].map(f.set_index("year").mean_load_MW)
        pred = np.polyval(c_rel, np.clip(d["T_degC"].values, t_lo, t_hi))
        r2 = 1 - np.sum((rel - pred) ** 2) / np.sum((rel - rel.mean()) ** 2)
        grid = np.polyval(c_rel, np.clip(T_GRID, t_lo, t_hi))
        xs = np.linspace(t_lo, t_hi, 400)
        ys = np.polyval(c_rel, xs)
        row = dict(region=r, years_used=", ".join(map(str, f.year)), n_years=len(f),
                   T_valid_min_degC=t_lo, T_valid_max_degC=t_hi)
        row.update({f"avg_{k}": v for k, v in zip(rel_cols, c_rel)})
        row.update(R2_on_all_years_rel=r2, T_at_min_load_degC=xs[np.argmin(ys)],
                   rel_load_at_T_valid_min=ys[0], rel_load_at_T_valid_max=ys[-1],
                   shape_flag=plausibility(Poly(DEGREE).fit(xs, ys), xs, ys))
        row.update({f"avg_{k}_MW": v for k, v in zip(a_cols, c_abs)})
        avg_rows.append(row)
        curves[r] = grid
    avg = pd.DataFrame(avg_rows)

    print()
    print(avg[["region", "years_used", "T_valid_min_degC", "T_valid_max_degC",
               "R2_on_all_years_rel", "T_at_min_load_degC"]].round(3).to_string(index=False))

    # ---- Excel
    notes = pd.DataFrame({"note": [
        "Daily load = mean of the 24 hourly loads of each UTC+2 calendar day; daily temperature = mean of "
        "the 24 hourly regional temperatures (deg C). One point per day.",
        f"Yearly fit: load(T) = a3*T^3 + a2*T^2 + a1*T + a0 (MW), least squares, degree {DEGREE} for every "
        "region and year. *_rel = coefficients / that year's mean daily load.",
        "Average cubic = mean of the yearly RELATIVE coefficients (identical to averaging the yearly "
        "curves point by point). avg_*_MW = mean of the MW coefficients, for reference only.",
        "Valid range: temperatures covered by every year used. Outside it, hold the curve flat at the "
        "edge value (the 'Average curve (rel)' sheet already does this).",
        "Normalisation: factor = f_rel(T_actual) / f_rel(T_normal), with f_rel the averaged relative cubic.",
        "R2_on_all_years_rel: how much of the day-to-day variation in relative load (all years) the "
        "averaged cubic explains.",
        "shape_flag: the curve bends back (load falls as it gets colder/hotter beyond its peak) by more "
        "than 1.5% of mean load. Consider holding it flat at its peak for colder/hotter days.",
        "Excluded: 2024 for A, B, D, E, F (exact/scaled copy of 2023) and national 2024; days "
        "2022-10-30 (left bank copied day), 2022-11-15 and 2022-11-23 (grid emergency/blackout).",
        "Load in A, B, D, E, F is NET of distributed PV (negative hours in 2025).",
        "Time basis: all series fixed UTC+2 (left bank converted from local time with DST - see 'Data fixes').",
    ]})
    path = OUT_DIR / "demand_temperature_cubic_2021_2025.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        avg.round(8).to_excel(xw, sheet_name="Average cubic", index=False)
        by_year.round(8).to_excel(xw, sheet_name="Cubic by year", index=False)
        pd.DataFrame(curves, index=pd.Index(T_GRID, name="T_degC")).round(5) \
            .to_excel(xw, sheet_name="Average curve (rel)")
        if comp_rows:
            comp = pd.concat(comp_rows)
            comp = comp[["region", "year"] + [c for c in comp.columns if c not in ("region", "year")]]
            comp.round(4).to_excel(xw, sheet_name="Model comparison", index=False)
        daily.round(3).to_excel(xw, sheet_name="Daily data")
        fixes.to_excel(xw, sheet_name="Data fixes", index=False)
        notes.to_excel(xw, sheet_name="Notes", index=False)
        for ws in xw.sheets.values():
            ws.freeze_panes = "B2"
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = \
                    max(10, min(60, len(str(col[0].value)) + 2))
    print(f"\nwritten {path}")

    # ---- chart: relative load, yearly cubics + average, per region
    year_colors = dict(zip(YEARS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]))
    n, ncol = len(regions), 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 4.4 * nrow), constrained_layout=True)
    for ax, r in zip(axes.ravel(), regions):
        f = by_year[(by_year.region == r) & (by_year.status == "used")]
        for _, row in f.iterrows():
            y = int(row.year)
            d = daily[(daily.region == r) & (daily.year == y) & daily.used_in_fit]
            ax.scatter(d["T_degC"], d["load_MW"] / row.mean_load_MW, s=5, color=year_colors[y],
                       alpha=0.25, linewidths=0)
            xs = np.linspace(row.T_min_degC, row.T_max_degC, 200)
            ax.plot(xs, np.polyval(row[rel_cols].values.astype(float), xs), color=year_colors[y],
                    lw=1.4, label=str(y))
        a = avg[avg.region == r]
        if not a.empty:
            a = a.iloc[0]
            xs = np.linspace(a.T_valid_min_degC, a.T_valid_max_degC, 200)
            c = a[[f"avg_{k}" for k in rel_cols]].values.astype(float)
            ax.plot(xs, np.polyval(c, xs), color="#0b0b0b", lw=2.8, label="average")
            ax.text(0.98, 0.97, f"years: {a.years_used}\nR² (all years) {a.R2_on_all_years_rel:.2f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#52514e")
        ax.axhline(1, color="#d0d0cc", lw=0.8, zorder=0)
        ax.set_title("National (sum)" if r == "NATIONAL" else r, loc="left", fontweight="bold")
        ax.set_xlabel("Daily average temperature (°C)", fontsize=9, color="#52514e")
        ax.set_ylabel("Daily load / year's mean load", fontsize=9, color="#52514e")
        ax.tick_params(labelsize=8, colors="#52514e")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#ecebe7", lw=0.8)
        ax.set_axisbelow(True)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    handles = {}
    for ax in axes.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    order = sorted(k for k in handles if k != "average") + (["average"] if "average" in handles else [])
    fig.legend([handles[k] for k in order], order, loc="upper right", ncol=6, frameon=False)
    fig.suptitle("Cubic fit of daily load vs temperature per year (relative to each year's mean) and their average",
                 x=0.01, ha="left", fontsize=13)
    png = OUT_DIR / "demand_temperature_cubic_2021_2025.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"written {png}")


if __name__ == "__main__":
    main()
