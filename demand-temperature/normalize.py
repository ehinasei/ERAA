"""
normalize.py
------------
Steps 2-4 of the USAID (2024) load time-series method (report p.34-36, Fig. 29-32),
per region: 7 regions x 16 climate years (2010-2025) of hourly load (8,760 h).

Uses the averaged relative cubic of each region from fit_demand_temperature.py:
    f(T) = a3*T^3 + a2*T^2 + a1*T + a0        (load relative to the year's mean)
held flat outside the temperature range covered by all fitted years, and held flat
at its peak where it would bend back (load must not fall as it gets colder/hotter).

1. Normal temperature (per region, per calendar day)
       average of that day's mean temperature over NORMAL_YEARS (2010-2025),
       smoothed with a centred 15-day moving average (wrapping around New Year).

2. Normalise the 2025 load (weather-neutral base profile)
       for each day d of 2025:  factor_d = f(T_normal,d) / f(T_2025,d)
       normalised load(h) = actual 2025 load(h) x factor_d   for all 24 hours of day d
   This removes the effect of 2025 being warmer/colder than normal on each day.
   The hourly shape within each day is kept.

3. Scale to the target annual energy
       scaled load = normalised load x (target GWh / normalised GWh)
   Target = the region's actual 2025 energy unless set in TARGET_ENERGY_GWH.

4. Climate years (put the temperature effect back)
       for climate year y and day d:  factor_d = f(T_y,d) / f(T_normal,d)
       load_y(h) = scaled load(h) x factor_d
   The 2025 calendar is kept (weekdays, holidays); only the weather changes, matched
   by calendar date. 29 February is dropped from leap climate years -> 8,760 hours.

Inputs (same folder as this script)
    Historical hourly electricity load.xlsx
    temperature_values/temperature_by_region_<year>.xlsx   2010-2025 (temp_by_region.py)
    fit_output/demand_temperature_cubic_2021_2025.xlsx     (fit_demand_temperature.py)

Output (normalized_output/)
    climate_year_load_2010_2025.xlsx
        Summary              per region x climate year: annual GWh, peak MW, peak hour
        <region> (7 tabs)    rows = 8,760 hours of the year (UTC+2, 2025 calendar),
                             columns = climate years 2010-2025 (MW)
        Base 2025            per region: actual, normalised and scaled 2025 load
        Normal temperature   per region and calendar day (deg C)
        Polynomials used     coefficients and valid temperature range per region
        Notes
    climate_year_peaks_2010_2025.png   peak load of each climate year per region (cf. Fig. 32)
    temperature_values/daily_temperature_by_region.csv   cache of daily temperatures
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent                    # demand-temperature
sys.path.insert(0, str(BASE))
# same load cleaning / time alignment as the fits
from fit_demand_temperature import (LOAD_FILE, TEMP_DIR, read_right_bank,   # noqa: E402
                                    read_left_bank, read_temperature, match_regions, u_shape)

# ---------------------------------------------------------------- settings
BASE_YEAR = 2025
CLIMATE_YEARS = range(2010, 2026)          # 2010 ... 2025
NORMAL_YEARS = range(2010, 2026)           # years averaged for the "normal" temperature
SMOOTH_DAYS = 15                           # centred moving average for the normal temperature
FIT_FILE = BASE / "fit_output" / "demand_temperature_cubic_2021_2025.xlsx"
OUT_DIR = BASE / "normalized_output"
DAILY_CACHE = TEMP_DIR / "daily_temperature_by_region.csv"

# Target annual energy per region (GWh). Leave a region out (or None) to keep its
# actual 2025 energy. Example: {"A": 520, "C": 2100}
TARGET_ENERGY_GWH = {}
# --------------------------------------------------------------------------


# ================================================================ inputs
def load_polynomials():
    p = pd.read_excel(FIT_FILE, sheet_name="Average cubic")
    p = p[p["region"] != "NATIONAL"].set_index("region")
    coef_cols = ["avg_a3_rel", "avg_a2_rel", "avg_a1_rel", "avg_a0_rel"]
    return p[coef_cols + ["T_valid_min_degC", "T_valid_max_degC", "years_used"]], coef_cols


def make_f(row, coef_cols):
    """f(T) for one region: averaged cubic, flat outside the valid range and beyond its peak."""
    lo, hi = row["T_valid_min_degC"], row["T_valid_max_degC"]
    grid = np.linspace(lo, hi, 2001)
    vals = u_shape(np.polyval(row[coef_cols].values.astype(float), grid))
    return lambda T: np.interp(np.clip(np.asarray(T, float), lo, hi), grid, vals)


def daily_temperatures(years):
    """Daily mean temperature (deg C) per region, from the yearly workbooks (cached)."""
    files = {y: TEMP_DIR / f"temperature_by_region_{y}.xlsx" for y in years}
    missing = [y for y, f in files.items() if not f.exists()]
    if missing:
        sys.exit(f"ERROR: temperature workbooks missing for {missing}. In temp_by_region.py set "
                 f"YEARS = range({min(years)}, {max(years) + 1}) and run it first.")
    newest = max(f.stat().st_mtime for f in files.values())
    if DAILY_CACHE.exists() and DAILY_CACHE.stat().st_mtime > newest:
        cached = pd.read_csv(DAILY_CACHE, index_col=0, parse_dates=True)
        if set(years) <= set(cached.index.year):
            return cached
    parts = []
    for y in years:
        print(f"reading temperature {y} ...")
        t = read_temperature(y)
        parts.append(t.resample("D").mean())
    daily = pd.concat(parts)
    daily.to_csv(DAILY_CACHE)
    return daily


def base_year_load():
    rb = pd.read_excel(LOAD_FILE, sheet_name="Right bank", header=None)
    lb = pd.read_excel(LOAD_FILE, sheet_name="Left bank", header=None)
    fixes = []
    load = read_right_bank(BASE_YEAR, rb, fixes)
    load = load[~load.index.duplicated()]
    load["LEFT"] = read_left_bank(BASE_YEAR, lb, fixes)
    full = pd.date_range(f"{BASE_YEAR}-01-01", f"{BASE_YEAR}-12-31 23:00", freq="h")
    return load.reindex(full).interpolate(limit=3)


def by_calendar_day(daily):
    """Index daily values by 'MM-DD' and drop 29 February."""
    d = daily[~((daily.index.month == 2) & (daily.index.day == 29))].copy()
    d.index = d.index.strftime("%m-%d")
    return d


# ================================================================ main
def main():
    OUT_DIR.mkdir(exist_ok=True)
    polys, coef_cols = load_polynomials()
    regions = list(polys.index)

    years_needed = sorted(set(CLIMATE_YEARS) | set(NORMAL_YEARS) | {BASE_YEAR})
    tdaily = daily_temperatures(years_needed)

    load = base_year_load()
    m = match_regions(list(load.columns), list(tdaily.columns))
    load = load.rename(columns=m)
    missing = [r for r in regions if r not in load.columns or r not in tdaily.columns]
    if missing:
        sys.exit(f"ERROR: no load or temperature for regions {missing}")
    load = load[regions]
    if load.isna().any().any():
        print("WARNING: gaps left in 2025 load after interpolation:", load.isna().sum().to_dict())

    # 1. normal temperature per calendar day (365 days), smoothed, wrapping around New Year
    normal_src = by_calendar_day(tdaily[tdaily.index.year.isin(NORMAL_YEARS)][regions])
    normal = normal_src.groupby(level=0).mean().sort_index()
    pad = SMOOTH_DAYS // 2
    wrapped = pd.concat([normal.iloc[-pad:], normal, normal.iloc[:pad]])
    normal = wrapped.rolling(SMOOTH_DAYS, center=True).mean().iloc[pad:-pad]

    fs = {r: make_f(polys.loc[r], coef_cols) for r in regions}
    day_key = load.index.strftime("%m-%d")                 # calendar day of every 2025 hour

    t_base = by_calendar_day(tdaily[tdaily.index.year == BASE_YEAR][regions])
    outputs, base_tabs, summary = {}, {}, []
    for r in regions:
        f = fs[r]
        f_normal = pd.Series(f(normal[r].values), index=normal.index)

        # 2. normalise 2025
        norm_factor = f_normal / pd.Series(f(t_base[r].values), index=t_base.index)
        actual = load[r]
        normalised = actual * norm_factor.reindex(day_key).values

        # 3. scale to target energy
        target = TARGET_ENERGY_GWH.get(r) or actual.sum() / 1000
        scaled = normalised * (target * 1000 / normalised.sum())
        base_tabs[r] = pd.DataFrame({"actual_MW": actual, "normalised_MW": normalised,
                                     "scaled_MW": scaled, "normalise_factor": norm_factor.reindex(day_key).values})

        # 4. climate years
        cols = {}
        for y in CLIMATE_YEARS:
            t_y = by_calendar_day(tdaily[tdaily.index.year == y][r])
            factor = pd.Series(f(t_y.values), index=t_y.index) / f_normal
            factor = factor.reindex(day_key).values
            if np.isnan(factor).any():
                print(f"WARNING: {r} {y}: {int(np.isnan(factor).sum() / 24)} day(s) without temperature "
                      f"-> normal temperature used")
                factor = np.where(np.isnan(factor), 1.0, factor)
            cols[y] = scaled.values * factor
            s = pd.Series(cols[y], index=load.index)
            summary.append(dict(region=r, climate_year=y, energy_GWh=s.sum() / 1000, peak_MW=s.max(),
                                peak_hour=s.idxmax().strftime("%m-%d %H:%M"), min_MW=s.min()))
        outputs[r] = pd.DataFrame(cols, index=pd.Index(load.index.strftime("%Y-%m-%d %H:%M"),
                                                        name=f"hour_{BASE_YEAR}_calendar_UTC+2"))
        print(f"{r:>14}: 2025 actual {actual.sum() / 1000:7.1f} GWh, peak {actual.max():6.1f} MW | "
              f"climate-year peaks {min(c.max() for c in cols.values()):6.1f}-"
              f"{max(c.max() for c in cols.values()):6.1f} MW")

    summ = pd.DataFrame(summary)

    # ---- Excel
    notes = pd.DataFrame({"note": [
        f"Base year {BASE_YEAR}; climate years {min(CLIMATE_YEARS)}-{max(CLIMATE_YEARS)}; normal temperature = "
        f"mean of {min(NORMAL_YEARS)}-{max(NORMAL_YEARS)} per calendar day, {SMOOTH_DAYS}-day centred smoothing.",
        "f(T) = averaged relative cubic per region (fit_demand_temperature.py), held flat outside its valid "
        "temperature range and at its peak where it would bend back.",
        "Normalise: load x f(T_normal)/f(T_2025) per day. Scale: to target GWh (default = actual 2025 GWh). "
        "Climate year y: scaled load x f(T_y)/f(T_normal) per day. Daily factor applied to all 24 hours.",
        "Rows follow the 2025 calendar (UTC+2, 8,760 h); weather is matched by calendar date; 29 Feb dropped.",
        "Load in A, B, D, E, F is NET of distributed PV (negative midday hours stay negative, scaled by the "
        "same daily factor). Dnestroenergo 2025 had an atypical year (no winter peak) - its base profile "
        "carries that shape into every climate year.",
        "Left bank converted from local time with DST to fixed UTC+2 (same as fit_demand_temperature.py).",
    ]})
    path = OUT_DIR / "climate_year_load_2010_2025.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        summ.round(2).to_excel(xw, sheet_name="Summary", index=False)
        for r in regions:
            outputs[r].round(3).to_excel(xw, sheet_name=str(r)[:31])
        base = pd.concat(base_tabs, axis=1)
        base.columns = [f"{r} {c}" for r, c in base.columns]
        base.index = base.index.strftime("%Y-%m-%d %H:%M")
        base.index.name = "hour_UTC+2"
        base.round(4).to_excel(xw, sheet_name="Base 2025")
        normal.round(3).rename_axis("calendar_day").to_excel(xw, sheet_name="Normal temperature")
        polys.to_excel(xw, sheet_name="Polynomials used")
        notes.to_excel(xw, sheet_name="Notes", index=False)
        for ws in xw.sheets.values():
            ws.freeze_panes = "B2"
            ws.column_dimensions["A"].width = 20
    print(f"\nwritten {path}")

    # ---- chart: peak of each climate year per region (cf. report Figure 32)
    n, ncol = len(regions), 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 3.8 * nrow), constrained_layout=True)
    for ax, r in zip(axes.ravel(), regions):
        s = summ[summ.region == r]
        ax.scatter(s.climate_year, s.peak_MW, s=30, color="#2a78d6")
        ax.axhline(load[r].max(), color="#eb6834", lw=1.2, ls="--", label="actual 2025 peak")
        ax.set_title(r, loc="left", fontweight="bold")
        ax.set_ylabel("Peak load (MW)", fontsize=9, color="#52514e")
        ax.tick_params(labelsize=8, colors="#52514e")
        ax.set_xticks(list(CLIMATE_YEARS)[::3])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#ecebe7", lw=0.8)
        ax.set_axisbelow(True)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    h, l = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", frameon=False)
    fig.suptitle(f"Peak load of each climate year ({BASE_YEAR} profile, {min(CLIMATE_YEARS)}-{max(CLIMATE_YEARS)} "
                 "weather), per region", x=0.01, ha="left", fontsize=13)
    png = OUT_DIR / "climate_year_peaks_2010_2025.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"written {png}")


if __name__ == "__main__":
    main()
 