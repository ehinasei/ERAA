"""
The purpose of script is to return hourly wind capacity factors (CF) for EVERY ERA5 grid cell, 2010-2026,
organised by region (A-F + Left Bank).

How cells are assigned to regions
  Each ERA5 cell is assigned to the region that covers the largest
  share of it. The map of regions has been approved by Moldovan Ministry of Energy. 
  If a region (ie Left Bank) is too small or narrow to be the largest share of any
  cell, it instead gets every cell that overlaps it (ie if even 5% is in, that cell is allocated to that region)
  Cells that don't touch any region are dropped.
 
 
Outputs:
  cf_wind_hourly_<year>.nc          hourly CF for all grid cells 
  wind_cf_by_region_<year>.xlsx     one workbook per year, one sheet per region:
                                    rows = hours, columns = grid cells in that region
  cell_region_mapping.csv           which cell belongs to which region (+ overlap share)
  cell_region_overlaps_all.csv      every cell-region overlap, for reference
  cell_region_map.png               map of the assignment, for checking
 
Notes
  - time_utc is the ERA5 time; time_local is fixed UTC+2 (no daylight saving).
  - Losses are applied (LOSS_FACTOR). 
  - Cell column names are the cell centre, e.g. 47.00N_28.75E.

"""

#IMPORTS
import os
 
import atlite 
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box
from pathlib import Path

#SETTINGS
BASE = Path(__file__).resolve().parent              
CUTOUT_DIR = BASE.parent / "cutouts"                 
OUT_DIR = BASE / "cf_output"                         
REGIONS_FILE = BASE.parent / "Moldova_ERAA_regions_full_partition.geojson" # approved regional map
REGION_FIELD = "region"               # name of the column holding the region name in GeoJSON;
                                  
YEARS = range(2010, 2027)          # 2010 ... 2026
TURBINE = "Vestas_V112_3MW"        # example turbine; can be changed to any atlite-supported turbine
LOSS_FACTOR = 0.88                 # wake, electrical and availability losses
LOCAL_TZ = "Etc/GMT-2"             # translates ERA5's UTC to Moldova's local time (UTC+2, no daylight saving)
MIN_OVERLAP = 0.0                  # keeps every cell that touches a region                             
EQUAL_AREA_CRS = "EPSG:3035"       # standard map projection mandated and used across EU for pan-European statistical analysis, environmental monitoring, and medium-scale mapping. Will be used to translate cells (defined by degrees) for AREA calculations (deg --> km). See https://spatialreference.org/ref/epsg/3035/


#HELPER FUNCTIONS
# calculates hourly CF for all grid cells for a given year. Saves as NetCDF. 
def compute_year(year):
    """Hourly CF for all grid cells of one year; saved as NetCDF"""
    source = os.path.join(CUTOUT_DIR, f"moldova_era5_{year}.nc") #filepath to read ERA5 data
    output = os.path.join(OUT_DIR, f"cf_wind_hourly_{year}.nc") #filepath to save hourly CF

    #using atlite to translate wind speeds to CFs 
    cutout = atlite.Cutout(source) #creates cutout from ERA5 data
    cf = cutout.wind(turbine=TURBINE,
                     interpolation_method="power",   # hourly shear exponent
                     smooth=True,                    # wind-farm power curve
                     aggregate_time=None)            # keep every hour (no sum/mean)
    
    cf = (cf * LOSS_FACTOR).astype("float32").rename("cf_wind") #adjusts CF with loss factor
    cf.attrs.update(units="capacity factor (0-1)", turbine=TURBINE,
                    loss_factor=LOSS_FACTOR, time_zone="UTC")
    cf.to_netcdf(output)
    print(f"{year}: hourly CF computed ({cf.sizes['time']} hours)")
    return output
 
 
# We need to sort CFs by region.
def load_regions():
    """Returns table of region names and their geometries, in CRS 4326 (lat/lon)"""
    regions = gpd.read_file(REGIONS_FILE) #reads GeoJSON into a table 
    regions = regions.set_crs(4326) if regions.crs is None else regions.to_crs(4326) #ensures that regions are in CRS 4326 (same as cells in ERA5)
 
    regions = regions[["region", "geometry"]] #extracts name of region and the geometry (polygon) of the region
    regions["region"] = regions["region"].astype(str).str.strip() #cleans region name so they are all strings
    regions = regions.dissolve(by="region").reset_index()     # merge multi-part regions
    print(f"Regions: {list(regions['region'])}")
    return regions #returns GeoDataFrame with 2 columns: region name and geometry (polygon)
 
 
def cell_polygons(x, y):
    """ERA5 data provides the coordinates of the grid cell center. 
        To figure out how much of a cell is in a given region, we need to draw the rest of the cell's boundaries."""
    dx = 0.25 #cells are 25 degrees by 25 degrees, as per ERA5 documentation
    dy = 0.25
    rows= []
    # x & y are lists of cell centre coordinates (longitudes and latitudes)
    for yi in y: #a value in list of cell latitudes (North-South)
        for xi in x: #a value in list of cell longitudes (East-West)
            rows.append({
                "y": yi,
                "x": xi, 
                "geometry": box(xi - dx / 2, yi - dy / 2, xi + dx / 2, yi + dy / 2)
            })
    cells= gpd.GeoDataFrame(rows, crs=4326) #returns GeoDataFrame with 270 cell coordinates. Each row of the DF has the cell center (y, x)and polygon shape. The DF has 3 columns.
    return cells
 
 #now that we have a table of regions with geographic boundaries and table of grid cells with geographic boundaries, we can figure out which cells belong to which regions.
def map_cells_to_regions(cells, regions):
    c = cells.to_crs(EQUAL_AREA_CRS) #.to_crs is a method from gpd. cells is the GeoDataFrame of ERA5 grid cells. 
    c["cell_area"] = c.area #calculates the area of each cell in km^2 and adds to as a new column to the GeoDataFrame.
    r = regions.to_crs(EQUAL_AREA_CRS) #converts the regions GeoDataFrame to the same CRS as the cells GeoDataFrame. 

    inter = gpd.overlay(c, r, how="intersection", keep_geom_type=True) #calculates the intersection of the two GeoDataFrames (cells and regions). Results in new GeoDF; 1 row/overlap. 5 columns: y, x, cell area, region, geometry (where y and x are coordinates of the cell's center, cell area is the area of the whole cell in km2, region where this cell overlaps, and geometry-- the overlapping piece with coordinates in meters)
    inter["overlap_frac"] = inter.area / inter["cell_area"] #calculates % cell area is in a region 
    inter = inter[inter["overlap_frac"] > 1e-6] #removes a cell where the overlap is less than 1e-6 (ie negligible)
 
    # each cell -> region with the largest overlap
    primary = inter.loc[inter.groupby(["y", "x"])["overlap_frac"].idxmax()] #assigns each cell to the region that it is the largest share of. returns dataframe with 1 row per cell (cells cannot be in 2 regions anymore)
    # idxmax() is a method in geopandas that returns the max value. 
    primary = primary[primary["overlap_frac"] >= MIN_OVERLAP] #keeps cells whose winning share (overlap fraction) is greater than 0. 
    mapping = primary[["region", "y", "x", "overlap_frac"]].assign(assignment="largest_overlap") #final GeoDF that will be used to map; 4 columns, including new assignment column

    names= []
    for yi, xi in zip(mapping["y"], mapping["x"]):
        names.append(f"{yi:.2f}N_{xi:.2f}E")
    mapping["cell_id"]= names #creates a list of cell names based on the coordinates of the cell center.

    mapping = mapping.sort_values(["region", "y", "x"], ascending=[True, False, True])
    overlaps = pd.DataFrame(inter[["region", "y", "x", "overlap_frac"]]) #saves all overlap fractions, just for reference
    return mapping.reset_index(drop=True), overlaps #returns the GeoDF and DF 


#plots regions and cells (indicating the cell center)
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
 
 
# Main function
def main():
    os.makedirs(OUT_DIR, exist_ok=True) #creates cf_output

    files = {}
    for y in YEARS:
         f= compute_year(y) #calculates CF for all grid cells for the year Y 
         if f: 
             files[y]= f
   
    #loads regions 
    regions = load_regions()
 
    #determines which cell belongs to which region 
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
        times = pd.DatetimeIndex(cf.time.values)
        local = times.tz_localize("UTC").tz_convert(LOCAL_TZ).strftime("%Y-%m-%d %H:%M")
        out = os.path.join(OUT_DIR, f"wind_cf_by_region_{year}.xlsx")
        with pd.ExcelWriter(out) as xw:
            for reg, grp in mapping.groupby("region"): #for each region, get the cells that belong to it 
                regional_cf = cf.sel(y=xr.DataArray(grp["y"].values, dims="cell"),
                             x=xr.DataArray(grp["x"].values, dims="cell"),
                             method="nearest")
                df = pd.DataFrame(regional_cf.transpose("time", "cell").values,
                                  index=times, columns=grp["cell_id"].values)
                df.index.name = "time_utc"
                df.insert(0, "time_local_UTC+2", local)
                df.round(4).to_excel(xw, sheet_name= reg)
        print(f"{year}: written {out}")
 
 
 
if __name__ == "__main__":
    main()
 