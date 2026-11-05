"""
The purpose of script is to return hourly solar capacity factors (CF) for EVERY ERA5 grid cell, 2010-2026,
organised by region (A-F + Left Bank)."""

from pathlib import Path
from assigning_cells_to_regions import run_regional
 
# ---------------------------------------------------------------- settings
BASE = Path(__file__).resolve().parent                 # RES availability curves
CUTOUT_DIR = BASE.parent / "cutouts"
OUT_DIR = BASE / "solar_cf_output"                     # solar results in their own folder
REGIONS_FILE = BASE.parent / "Moldova_ERAA_regions_full_partition.geojson"
YEARS = range(2010, 2027)
LOCAL_TZ = "Etc/GMT-2"                                 # fixed UTC+2
MIN_OVERLAP = 0.0
 
PANEL = "CSi"                       # crystalline silicon, Huld (2010) model
ORIENTATION = "latitude_optimal"    # fixed, south-facing, optimal tilt
TRACKING = None                     # None = fixed; "horizontal" = single-axis
LOSS_FACTOR = 0.97                  # soiling, wiring, downtime (inverter 0.9 is
                                    # already applied inside atlite's CSi panel)
# --------------------------------------------------------------------------
 
 
def solar_cf(cutout):
    """Hourly solar PV CF for every grid cell of one cutout."""
    cf = cutout.pv(panel=PANEL,
                   orientation=ORIENTATION,
                   tracking=TRACKING,
                   aggregate_time=None)                # keep every hour
    return cf * LOSS_FACTOR
 
 
if __name__ == "__main__":
    run_regional(make_cf=solar_cf, name="solar", years=YEARS,
                 cutout_dir=CUTOUT_DIR, out_dir=OUT_DIR, regions_file=REGIONS_FILE,
                 attrs={"panel": PANEL, "orientation": str(ORIENTATION),
                        "tracking": str(TRACKING), "loss_factor": LOSS_FACTOR})
 