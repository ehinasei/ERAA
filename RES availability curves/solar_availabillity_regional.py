
"""
Hourly solar PV capacity factors (CF) for EVERY ERA5 grid cell, 2010-2026,
organised by region (A-F + Left Bank). Nothing is averaged: each grid cell
keeps its own hourly series.
 
How cells are assigned to regions
  Each ERA5 cell (~28 x 19 km) is assigned to the region that covers the largest
  share of it. If a region is too small or narrow to be the largest share of any
  cell (possible for the Left Bank), it instead gets every cell that overlaps it,
  and the script prints a warning. Cells that don't touch any region are dropped.
  Check the result in cell_region_map.png.
 
Outputs (in OUT_DIR):
  cf_solar_hourly_<year>.nc          hourly CF for all grid cells (reused if already there)
  solar_cf_by_region_<year>.xlsx     one workbook per year, one sheet per region:
                                    rows = hours, columns = grid cells in that region
  cell_region_mapping.csv           which cell belongs to which region (+ overlap share)
  cell_region_overlaps_all.csv      every cell-region overlap, for reference
  cell_region_map.png               map of the assignment, for checking
 
Notes
  - time_utc is the ERA5 time; time_local is fixed UTC+2 (no daylight saving).
  - Losses are applied (LOSS_FACTOR).
  - Night hours are 0. Sanity check: the daily peak should be around 12:00-13:00
    in the time_local column; if it is shifted, the radiation timing needs checking.
  - CF is relative to the panels' rated (DC) capacity. If LEAP capacity is in AC
    with a DC/AC ratio > 1, the CF relative to AC capacity will be somewhat higher.
  - Cell column names are the cell centre, e.g. 47.00N_28.75E.
 
"""
 
import os
import re
 
import atlite
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box
 
# ---------------------------------------------------------------- settings
CUTOUT_DIR = "cutouts"
OUT_DIR = "cf_output"
REGIONS_FILE = "regions.geojson"   # your GeoJSON with regions A-F + Left Bank
REGION_FIELD = None                # name of the column holding the region name;
                                   
YEARS = range(2010, 2027)          # 2010 ... 2026
PANEL = "CSi"                      # crystalline silicon (see atlite.solarpanels)
ORIENTATION = "latitude_optimal"   # fixed tilt at the optimal angle, facing south;
                                   # or e.g. {"slope": 30, "azimuth": 180}
TRACKING = None                    # None = fixed; "horizontal" = single-axis tracking
LOSS_FACTOR = 0.90                 # soiling, wiring, mismatch, downtime; check whether
                                  
LOCAL_TZ = "Etc/GMT-2"             # fixed UTC+2 for the time_local column
MIN_OVERLAP = 0.0                  # drop cells whose share inside their region is
                                   # below this (e.g. 0.1 = 10%); 0 keeps all
EQUAL_AREA_CRS = "EPSG:3035"       # equal-area projection for measuring overlaps
# --------------------------------------------------------------------------
 
 
# ---------------------------------------------------------------- hourly CF per year
def compute_year(year):
    """Hourly CF for all grid cells of one year; saved as NetCDF. Returns path or None."""
    src = os.path.join(CUTOUT_DIR, f"moldova_era5_{year}.nc")
    out = os.path.join(OUT_DIR, f"cf_solar_hourly_{year}.nc")
    if os.path.exists(out):
        with xr.open_dataarray(out) as old:
            ok = "time" in old.dims and old.sizes["time"] > 1000
        if ok:
            return out
        os.remove(out)          # file from an earlier run without hourly data
        print(f"{year}: removed old non-hourly file, recomputing")
    if not os.path.exists(src):
        print(f"{year}: no cutout found - skipping")
        return None
    cutout = atlite.Cutout(src)

    #calculating capacity factor using atlite's pv method 
    cf = cutout.pv(panel=PANEL,
                   orientation=ORIENTATION,
                   tracking=TRACKING,
                   aggregate_time=None)              # keep every hour (no sum/mean)
    if "time" not in cf.dims:
        raise RuntimeError("atlite returned no time dimension - check aggregate_time=None")
    cf = (cf * LOSS_FACTOR).astype("float32").rename("cf_solar")
    cf.attrs.update(units="capacity factor (0-1)", panel=PANEL, orientation=str(ORIENTATION),
                    tracking=str(TRACKING),
                    loss_factor=LOSS_FACTOR, bias_corrected="no", time_zone="UTC")
    cf.to_netcdf(out)
    print(f"{year}: hourly CF computed ({cf.sizes['time']} hours)")
    return out
 
 
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
    regions = regions.dissolve(by="region").reset_index()     # merge multi-part regions
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
 
 
def map_cells_to_regions(cells, regions):
    c = cells.to_crs(EQUAL_AREA_CRS)
    c["cell_area"] = c.area
    r = regions.to_crs(EQUAL_AREA_CRS)
    inter = gpd.overlay(c, r, how="intersection", keep_geom_type=True)
    inter["overlap_frac"] = inter.area / inter["cell_area"]
    inter = inter[inter["overlap_frac"] > 1e-6]
 
    # each cell -> region with the largest overlap
    primary = inter.loc[inter.groupby(["y", "x"])["overlap_frac"].idxmax()]
    primary = primary[primary["overlap_frac"] >= MIN_OVERLAP]
    mapping = primary[["region", "y", "x", "overlap_frac"]].assign(assignment="largest_overlap")
 
    # regions that didn't get any cell -> give them every cell that overlaps them
    for reg in sorted(set(regions["region"]) - set(mapping["region"])):
        extra = inter[inter["region"] == reg][["region", "y", "x", "overlap_frac"]]
        mapping = pd.concat([mapping, extra.assign(assignment="any_overlap")])
        print(f"WARNING: region '{reg}' is smaller/narrower than the ERA5 cells. "
              f"It was given all {len(extra)} cells that overlap it; these cells "
              f"also appear under a neighbouring region.")
 
    mapping["cell_id"] = [f"{yy:.2f}N_{xx:.2f}E" for yy, xx in zip(mapping["y"], mapping["x"])]
    mapping = mapping.sort_values(["region", "y", "x"], ascending=[True, False, True])
    overlaps = pd.DataFrame(inter[["region", "y", "x", "overlap_frac"]])
    return mapping.reset_index(drop=True), overlaps
 
 
def plot_mapping(cells, regions, mapping, path):
    fig, ax = plt.subplots(figsize=(8, 8))
    cells.boundary.plot(ax=ax, color="lightgrey", linewidth=0.4)
    regions.boundary.plot(ax=ax, color="black", linewidth=1)
    for reg, grp in mapping.groupby("region"):
        ax.scatter(grp["x"], grp["y"], s=40, label=f"{reg} ({len(grp)} cells)")
    for _, row in regions.iterrows():
        p = row.geometry.representative_point()
        ax.annotate(row["region"], (p.x, p.y), fontsize=11, weight="bold", ha="center")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("ERA5 grid cells (dots = cell centres) assigned to regions")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
 
 
def safe_sheet_name(name):
    return re.sub(r"[\[\]\*\?/\\:]", "_", name)[:31]
 
 
# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
 
    files = {y: f for y in YEARS if (f := compute_year(y))}
    if not files:
        print("No cutouts found - check CUTOUT_DIR.")
        return
 
    regions = load_regions()
 
    # build the cell -> region mapping from the first year's grid
    with xr.open_dataarray(next(iter(files.values()))) as da:
        grid_x, grid_y = da.x.values, da.y.values
    cells = cell_polygons(grid_x, grid_y)
    mapping, overlaps = map_cells_to_regions(cells, regions)
    mapping.to_csv(os.path.join(OUT_DIR, "cell_region_mapping.csv"), index=False)
    overlaps.to_csv(os.path.join(OUT_DIR, "cell_region_overlaps_all.csv"), index=False)
    plot_mapping(cells, regions, mapping, os.path.join(OUT_DIR, "cell_region_map.png"))
    print("\nCells per region:")
    print(mapping.groupby("region").size().to_string())
 
    # one workbook per year, one sheet per region, one column per grid cell
    for year, f in files.items():
        with xr.open_dataarray(f) as da:
            cf = da.load()
        if not (np.array_equal(cf.x.values, grid_x) and np.array_equal(cf.y.values, grid_y)):
            print(f"{year}: grid differs from the other years - skipped "
                  f"(was this cutout downloaded with different boundaries?)")
            continue
 
        times = pd.DatetimeIndex(cf.time.values)
        local = times.tz_localize("UTC").tz_convert(LOCAL_TZ).strftime("%Y-%m-%d %H:%M")
        out = os.path.join(OUT_DIR, f"solar_cf_by_region_{year}.xlsx")
        with pd.ExcelWriter(out) as xw:
            for reg, grp in mapping.groupby("region"):
                sel = cf.sel(y=xr.DataArray(grp["y"].values, dims="cell"),
                             x=xr.DataArray(grp["x"].values, dims="cell"),
                             method="nearest")
                df = pd.DataFrame(sel.transpose("time", "cell").values,
                                  index=times, columns=grp["cell_id"].values)
                df.index.name = "time_utc"
                df.insert(0, "time_local_UTC+2", local)
                df.round(4).to_excel(xw, sheet_name=safe_sheet_name(reg))
        print(f"{year}: written {out}")
 
    print(f"\nDone. Check {OUT_DIR}/cell_region_map.png to confirm the cell assignment.")
 
 
if __name__ == "__main__":
    main()
 