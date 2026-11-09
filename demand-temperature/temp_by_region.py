""" 
The purpose of this script is to sort hourly temperature data by region and year for climate years 2021-2025. 

This file returns the hourly temperature for every ERA5 grid cell, organised by region.
Output (in OUT_DIR):
    temperature_by_region_<year>.xlsx   one workbook per year (2021-2025)
    cell_region_mapping.csv            
    cell_region_map.png                 

Cells are assigned to regions exactly as for wind and solar (largest overlap),
Temperatures are ERA5 2 m air temperature in Kelvin, read straight from the cutouts.
region_mean_K = average of the region's cells for that hour 
ERA5 data is in UTC. Historical hourly electricity demand data is in Moldovan local time (UTC+2) """

#IMPORTS
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

BASE = Path(__file__).resolve().parent                 # demand-temperature
sys.path.insert(0, str(BASE.parent / "RES availability curves"))
from assigning_cells_to_regions import (load_regions, cell_polygons, 
                                        map_cells_to_regions, plot_mapping)

#SETTINGS
YEARS = range(2010, 2026)                              # 2010 ... 2025
CUTOUT_DIR = BASE.parent / "cutouts"
OUT_DIR = BASE / "temperature_values"                  # temperature results in their own folder
REGIONS_FILE = BASE.parent / "Moldova_ERAA_regions_full_partition.geojson"
LOCAL_TZ = "Etc/GMT-2"                                 # fixed UTC+2


#HELPER FUNCTIONS
def open_temperature(year):
    """Hourly 2 m temperature (K) for every grid cell of one cutout year.
    The cutouts were prepared with atlite's "temperature" feature, so ERA5 2 m
    temperature is stored in them directly, in Kelvin (variable 'temperature')."""
    path = CUTOUT_DIR / f"moldova_era5_{year}.nc"
    if not path.exists():
        return None
    return xr.open_dataset(path)["temperature"].transpose("time", "y", "x")
    
def local_year(year):
    """Temperature for the LOCAL year (UTC+2), stitched from this and the previous cutout."""
    start = pd.Timestamp(f"{year}-01-01", tz=LOCAL_TZ).tz_convert("UTC").tz_localize(None)
    end = pd.Timestamp(f"{year + 1}-01-01", tz=LOCAL_TZ).tz_convert("UTC").tz_localize(None)
    last = end - pd.Timedelta("1s")

    cur = open_temperature(year)
    if cur is None:
        print(f"{year}: no cutout found - skipped")
        return None
    parts = [cur.sel(time=slice(start, last))]
    prev = open_temperature(year - 1)
    if prev is not None:
        parts.insert(0, prev.sel(time=slice(start, last)))
    t = xr.concat(parts, dim="time").load() if len(parts) > 1 else parts[0].load()
    t = t.isel(time=~pd.Index(t.time.values).duplicated())

    full = pd.date_range(start, end, freq="h", inclusive="left")
    missing = len(full.difference(pd.DatetimeIndex(t.time.values)))
    if missing:
        print(f"{year}: {missing} hour(s) missing (left blank)")
    return t.reindex(time=full)


#MAIN
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    first = next((y for y in YEARS if open_temperature(y) is not None), None)
    if first is None:
        print(f"No cutouts found in {CUTOUT_DIR}")
        return
    grid = open_temperature(first)
    grid_x, grid_y = grid.x.values, grid.y.values
    regions = load_regions(REGIONS_FILE)
    cells = cell_polygons(grid_x, grid_y)
    mapping, _ = map_cells_to_regions(cells, regions)
    mapping.to_csv(OUT_DIR / "cell_region_mapping.csv", index=False)
    plot_mapping(cells, regions, mapping, OUT_DIR / "cell_region_map.png")
    print("\nCells per region:")
    print(mapping.groupby("region").size().to_string())

    # one workbook per year, one tab per region, one column per grid cell
    for year in YEARS:
        t = local_year(year)
        if t is None:
            continue
        if not (np.array_equal(t.x.values, grid_x) and np.array_equal(t.y.values, grid_y)):
            print(f"{year}: grid differs from the other years - skipped")
            continue
        times_utc = pd.DatetimeIndex(t.time.values)
        local = times_utc.tz_localize("UTC").tz_convert(LOCAL_TZ).strftime("%Y-%m-%d %H:%M")

        out = OUT_DIR / f"temperature_by_region_{year}.xlsx"
        with pd.ExcelWriter(out) as xw:
            for reg, grp in mapping.groupby("region"):        # cells that belong to this region
                regional_t = t.sel(y=xr.DataArray(grp["y"].values, dims="cell"),
                                   x=xr.DataArray(grp["x"].values, dims="cell"),
                                   method="nearest")
                df = pd.DataFrame(regional_t.transpose("time", "cell").values,
                                  index=pd.Index(local, name="time_local_UTC+2"),
                                  columns=grp["cell_id"].values)
                df.insert(0, "region_mean_K", df.mean(axis=1))
                df.insert(0, "time_utc", times_utc.strftime("%Y-%m-%d %H:%M"))
                df.round(3).to_excel(xw, sheet_name=str(reg)[:31])
        print(f"{year}: written {out.name} ({len(times_utc)} hours)")


if __name__ == "__main__":
    main()
 