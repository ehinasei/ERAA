"""
The purpose of script is to return hourly wind capacity factors (CF) for EVERY ERA5 grid cell, 2010-2026,
organised by region (A-F + Left Bank)."""


from pathlib import Path 
from assigning_cells_to_regions import run_regional

# ---------------------------------------------------------------- settings
BASE = Path(__file__).resolve().parent                 # RES availability curves
CUTOUT_DIR = BASE.parent / "cutouts"
OUT_DIR = BASE / "windcf_output"
REGIONS_FILE = BASE.parent / "Moldova_ERAA_regions_full_partition.geojson"
YEARS = range(2010, 2027)
LOCAL_TZ = "Etc/GMT-2"                                 # fixed UTC+2
MIN_OVERLAP = 0.0
 
TURBINE = "Vestas_V112_3MW" #moldova's wind turbines are primarily second-hand, medium-capacity horizontal-axis turbines)
LOSS_FACTOR = 0.88                                     # wake, electrical, downtime
# --------------------------------------------------------------------------
 
 
def wind_cf(cutout):
    """Hourly wind CF for every grid cell of one cutout."""
    cf = cutout.wind(turbine=TURBINE,
                     interpolation_method="power",     # hourly shear exponent
                     smooth=True,                      # wind-farm power curve
                     aggregate_time=None)              # keep every hour
    return cf * LOSS_FACTOR
 
 
if __name__ == "__main__":
    run_regional(make_cf=wind_cf, name="wind", years=YEARS,
                 cutout_dir=CUTOUT_DIR, out_dir=OUT_DIR, regions_file=REGIONS_FILE,
                 attrs={"turbine": TURBINE, "loss_factor": LOSS_FACTOR})
 