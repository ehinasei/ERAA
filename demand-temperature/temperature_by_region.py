"""
Hourly 2 m temperature by region (A-F + Left Bank) for a given climate year,
from the same ERA5 cutouts used for the wind capacity factors.
 
Unlike the wind script, temperature IS averaged: each region gets ONE hourly
series, so the output is a single sheet with one column per region.
 
How cells are combined into a regional temperature
  Every ERA5 cell (~28 x 19 km) that overlaps a region contributes to that
  region's average, weighted by the area of the overlap. A cell on a border is
  therefore split between the regions it touches, and small or narrow regions
  (e.g. the Left Bank) still get a proper average instead of borrowing whole
  cells from their neighbours. A NATIONAL column (all regions together, same
  area weighting) is added for fitting the national load-temperature curve.
  Check the assignment in cell_region_map.png.
 
Outputs (in OUT_DIR):
  temperature_by_region_<year>.xlsx  ONE sheet: rows = hours of the climate year,
                                     columns = time_utc, time_local, one column per
                                     region, NATIONAL. Values in deg C.
  cell_region_weights.csv            every cell-region overlap and its weight
  cell_region_map.png                map of the assignment, for checking
 
Notes
  - The climate year is a LOCAL year (UTC+2): 1 Jan 00:00 to 31 Dec 23:00 local.
    Its first 2 hours fall in the previous UTC year, so the previous year's
    cutout is read too when available (otherwise those hours are left blank).
  - time_local is fixed UTC+2 (no daylight saving), as in the wind script.
  - Leap years have 8784 hours. Set DROP_FEB29 = True to get 8760 for LEAP.
  - Reads the 'temperature' variable (2 m air temperature, Kelvin) that atlite
    stores in the cutout; 
Usage
  python temperature_by_region.py              # years in CLIMATE_YEARS below
  python temperature_by_region.py 2018         # one year
  python temperature_by_region.py 2010-2026    # a range
"""
 
import os
import sys
 
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box
 
# ---------------------------------------------------------------- settings
CUTOUT_DIR = "cutouts"
OUT_DIR = "temp_output"
REGIONS_FILE = "regions_not_raions.geojson"   # your GeoJSON with regions A-F + Left Bank
REGION_FIELD = None                # name of the column holding the region name;
                                   # None = detect automatically
CLIMATE_YEARS = [2025]             # default if no year is given on the command line
 
TEMP_VAR = "temperature"           # variable name inside the atlite cutout
LOCAL_TZ = "Etc/GMT-2"             # fixed UTC+2 ('Etc/GMT-2' means UTC PLUS 2)
ADD_NATIONAL = True                # add a NATIONAL column (all regions together)
DROP_FEB29 = False                 # True -> 8760 hours in leap years too
EQUAL_AREA_CRS = "EPSG:3035"       # equal-area projection for measuring overlaps
NATIONAL = "NATIONAL"
# --------------------------------------------------------------------------
 
 
# ---------------------------------------------------------------- reading cutouts
def cutout_path(year):
    return os.path.join(CUTOUT_DIR, f"moldova_era5_{year}.nc")
 
 
def open_temperature(year):
    """Lazy DataArray (time, y, x) of 2 m temperature for one cutout year, or None."""
    path = cutout_path(year)
    if not os.path.exists(path):
        return None
    ds = xr.open_dataset(path)
    if TEMP_VAR not in ds:
        raise KeyError(f"'{TEMP_VAR}' not in {path}. Variables are: {list(ds.data_vars)}. "
                       f"The cutout may have been prepared without the 'temperature' "
                       f"feature (re-run cutout.prepare(features=['temperature'])).")
    return ds[TEMP_VAR].transpose("time", "y", "x")
 
 
def local_year_utc_bounds(year):
    """UTC start (inclusive) and end (exclusive) of the local climate year."""
    start = pd.Timestamp(f"{year}-01-01 00:00", tz=LOCAL_TZ).tz_convert("UTC").tz_localize(None)
    end = pd.Timestamp(f"{year + 1}-01-01 00:00", tz=LOCAL_TZ).tz_convert("UTC").tz_localize(None)
    return start, end
 
 
def load_climate_year(year):
    """Hourly temperature for the local climate year, stitched from two cutouts if needed."""
    cur = open_temperature(year)
    if cur is None:
        print(f"{year}: no cutout found ({cutout_path(year)}) - skipping")
        return None
    start, end = local_year_utc_bounds(year)
    parts = []
    prev = open_temperature(year - 1)
    if prev is not None:
        if not (np.array_equal(prev.x.values, cur.x.values)
                and np.array_equal(prev.y.values, cur.y.values)):
            print(f"{year}: grid of {year - 1} cutout differs - first hours left blank")
        else:
            parts.append(prev.sel(time=slice(start, end - pd.Timedelta("1s"))))
    parts.append(cur.sel(time=slice(start, end - pd.Timedelta("1s"))))
    da = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
    da = da.load()
    da = da.isel(time=~pd.Index(da.time.values).duplicated())
 
    # Put on a complete hourly axis so missing hours show up as blanks
    full = pd.date_range(start, end, freq="h", inclusive="left")
    missing = len(full.difference(pd.DatetimeIndex(da.time.values)))
    da = da.reindex(time=full)
    if missing:
        print(f"{year}: {missing} hour(s) missing from the cutouts (left blank)")
    return da
 
 
# ---------------------------------------------------------------- regions
def load_regions():
    regions = gpd.read_file(REGIONS_FILE)
    regions = regions.set_crs(4326) if regions.crs is None else regions.to_crs(4326)
 
    field = REGION_FIELD
    if field is None:
        cols = [c for c in regions.columns if c != "geometry"]
        guesses = [c for c in cols if c.lower() in ("region", "zone", "name", "id", "label")]
        if len(cols) == 1:
            field = cols[0]
        elif guesses:
            field = guesses[0]
        else:
            raise ValueError(f"Can't tell which column holds the region names. "
                             f"Set REGION_FIELD to one of: {cols}")
    regions = regions[[field, "geometry"]].rename(columns={field: "region"})
    regions["region"] = regions["region"].astype(str).str.strip()
    regions["geometry"] = regions.geometry.buffer(0)            # fix invalid polygons
    regions = regions.dissolve(by="region").reset_index()       # merge multi-part regions
    print(f"Regions (from column '{field}'): {list(regions['region'])}")
    return regions
 
 
def cell_polygons(x, y):
    """One box per ERA5 grid cell, centred on the cell's coordinates."""
    dx = float(np.abs(np.diff(x)).mean())
    dy = float(np.abs(np.diff(y)).mean())
    rows = [{"y": float(yi), "x": float(xi),
             "geometry": box(xi - dx / 2, yi - dy / 2, xi + dx / 2, yi + dy / 2)}
            for yi in y for xi in x]
    return gpd.GeoDataFrame(rows, crs=4326)
 
 
def cell_weights(cells, regions):
    """Area of every cell-region overlap -> weights that sum to 1 within each region."""
    c = cells.to_crs(EQUAL_AREA_CRS)
    c["cell_area_km2"] = c.area / 1e6
    r = regions.to_crs(EQUAL_AREA_CRS)
    inter = gpd.overlay(c, r, how="intersection", keep_geom_type=True)
    inter["overlap_km2"] = inter.area / 1e6
    inter["overlap_frac_of_cell"] = inter["overlap_km2"] / inter["cell_area_km2"]
    w = pd.DataFrame(inter[["region", "y", "x", "overlap_km2", "overlap_frac_of_cell"]])
    w = w[w["overlap_km2"] > 1e-3]
 
    if ADD_NATIONAL:
        nat = w.groupby(["y", "x"], as_index=False).agg(
            overlap_km2=("overlap_km2", "sum"), overlap_frac_of_cell=("overlap_frac_of_cell", "sum"))
        nat["region"] = NATIONAL
        w = pd.concat([w, nat[w.columns]], ignore_index=True)
 
    w["weight"] = w["overlap_km2"] / w.groupby("region")["overlap_km2"].transform("sum")
    w["cell_id"] = [f"{yy:.2f}N_{xx:.2f}E" for yy, xx in zip(w["y"], w["x"])]
    w = w.sort_values(["region", "y", "x"], ascending=[True, False, True]).reset_index(drop=True)
 
    missing = set(regions["region"]) - set(w["region"])
    if missing:
        print(f"WARNING: no ERA5 cell overlaps region(s) {missing} - check the cutout extent.")
    return w
 
 
def plot_mapping(cells, regions, w, path):
    fig, ax = plt.subplots(figsize=(8, 8))
    cells.boundary.plot(ax=ax, color="lightgrey", linewidth=0.4)
    regions.boundary.plot(ax=ax, color="black", linewidth=1)
    ww = w[w["region"] != NATIONAL]
    for reg, grp in ww.groupby("region"):
        ax.scatter(grp["x"], grp["y"], s=20 + 300 * grp["weight"], alpha=0.6,
                   label=f"{reg} ({len(grp)} cells)")
    for _, row in regions.iterrows():
        p = row.geometry.representative_point()
        ax.annotate(row["region"], (p.x, p.y), fontsize=11, weight="bold", ha="center")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("ERA5 cells contributing to each region\n(dot size = weight in that region)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
 
 
# ---------------------------------------------------------------- averaging
def regional_average(da, w):
    """Area-weighted mean per region. NaN cells are skipped and the weights renormalised."""
    regions = [r for r in w["region"].unique() if r != NATIONAL]
    regions = sorted(regions) + ([NATIONAL] if ADD_NATIONAL else [])
    cells = w[["y", "x"]].drop_duplicates().reset_index(drop=True)
    iy = pd.Index(da.y.values).get_indexer(cells["y"])
    ix = pd.Index(da.x.values).get_indexer(cells["x"])
    if (iy < 0).any() or (ix < 0).any():
        raise ValueError("Some weighted cells are not on this year's grid.")
 
    W = np.zeros((len(cells), len(regions)))
    k = w.merge(cells.reset_index().rename(columns={"index": "k"}), on=["y", "x"])
    W[k["k"].values, k["region"].map({r: i for i, r in enumerate(regions)}).values] = k["weight"].values
 
    X = da.values[:, iy, ix].astype("float64")          # (time, cells)
    valid = ~np.isnan(X)
    num = np.where(valid, X, 0.0) @ W
    den = valid.astype(float) @ W
    with np.errstate(invalid="ignore", divide="ignore"):
        T = np.where(den > 0, num / den, np.nan)
    if np.nanmedian(T) > 150:                           # Kelvin -> deg C
        T = T - 273.15
    return pd.DataFrame(T, index=pd.DatetimeIndex(da.time.values), columns=regions)
 
 
# ---------------------------------------------------------------- output
def write_year(year, df):
    times_utc = df.index
    local = times_utc.tz_localize("UTC").tz_convert(LOCAL_TZ)
    if DROP_FEB29:
        keep = ~((local.month == 2) & (local.day == 29))
        df, times_utc, local = df[keep], times_utc[keep], local[keep]
 
    out_df = df.round(2).copy()
    out_df.insert(0, "time_local_UTC+2", local.strftime("%Y-%m-%d %H:%M"))
    out_df.insert(0, "time_utc", times_utc.strftime("%Y-%m-%d %H:%M"))
    out_df.columns = list(out_df.columns[:2]) + [f"{c} (degC)" for c in out_df.columns[2:]]
 
    out = os.path.join(OUT_DIR, f"temperature_by_region_{year}.xlsx")
    sheet = f"temperature_{year}"
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        out_df.to_excel(xw, sheet_name=sheet, index=False)
        ws = xw.sheets[sheet]
        ws.freeze_panes = "C2"
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = max(12, len(str(col[0].value)) + 2)
 
    blanks = int(df.isna().any(axis=1).sum())
    print(f"{year}: written {out}  ({len(df)} hours"
          + (f", {blanks} with blanks" if blanks else "") + ")")
    print("   annual mean / min / max (degC): "
          + "; ".join(f"{c} {df[c].mean():.1f}/{df[c].min():.1f}/{df[c].max():.1f}"
                      for c in df.columns))
 
 
# ---------------------------------------------------------------- main
def parse_years(argv):
    if len(argv) < 2:
        return list(CLIMATE_YEARS)
    years = []
    for a in argv[1:]:
        if "-" in a:
            lo, hi = a.split("-")
            years += list(range(int(lo), int(hi) + 1))
        else:
            years.append(int(a))
    return years
 
 
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    years = parse_years(sys.argv)
 
    first = next((y for y in years if os.path.exists(cutout_path(y))), None)
    if first is None:
        print(f"No cutouts found for {years} in {CUTOUT_DIR}.")
        return
 
    regions = load_regions()
    grid = open_temperature(first)
    grid_x, grid_y = grid.x.values, grid.y.values
    cells = cell_polygons(grid_x, grid_y)
    w = cell_weights(cells, regions)
    w.to_csv(os.path.join(OUT_DIR, "cell_region_weights.csv"), index=False)
    plot_mapping(cells, regions, w, os.path.join(OUT_DIR, "cell_region_map.png"))
    print("\nCells contributing to each region:")
    print(w.groupby("region").size().to_string())
 
    for year in years:
        da = load_climate_year(year)
        if da is None:
            continue
        if not (np.array_equal(da.x.values, grid_x) and np.array_equal(da.y.values, grid_y)):
            print(f"{year}: grid differs from the other years - skipped "
                  f"(was this cutout downloaded with different boundaries?)")
            continue
        write_year(year, regional_average(da, w))
 
    print(f"\nDone. Check {OUT_DIR}/cell_region_map.png to confirm the cell assignment.")
 
 
if __name__ == "__main__":
    main()