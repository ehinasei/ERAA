
import logging
import os
import shutil
 
import atlite
import pandas as pd
import xarray as xr
 
# ---------------------------------------------------------------- settings
OUT_DIR = "cutouts"
YEARS = [y for y in range(2010, 2027) if y != 2024]   # 2024 comes from the test                  # 2010 ... 2026
LAST_DAY_2026 = "2026-09-25"                 # ERA5 lags ~5 days behind real time
X = slice(26.25, 30.5)                       # longitude (must match the 2024 test)
Y = slice(45.25, 48.75)                      # latitude
FEATURES = ["wind", "influx", "temperature"]
TEST_FILE = "moldova_era5_2024_test.nc"      # your earlier 2024 download
ATTEMPTS_PER_YEAR = 2
# --------------------------------------------------------------------------
 
log = logging.getLogger("era5")
 
 
def path_for(year):
    return os.path.join(OUT_DIR, f"moldova_era5_{year}.nc")
 
 
def end_date(year):
    return LAST_DAY_2026 if year == 2026 else f"{year}-12-31"
 
 
def is_complete(path, year):
    """Check a cutout file covers the whole year on the expected grid."""
    if not os.path.exists(path):
        return False
    try:
        with xr.open_dataset(path) as ds:
            expected = len(pd.date_range(f"{year}-01-01 00:00",
                                         f"{end_date(year)} 23:00", freq="h"))
            n_time = ds.sizes.get("time", 0)
            half_cell = 0.13
            grid_ok = (abs(float(ds.x.min()) - X.start) < half_cell and
                       abs(float(ds.x.max()) - X.stop) < half_cell and
                       abs(float(ds.y.min()) - Y.start) < half_cell and
                       abs(float(ds.y.max()) - Y.stop) < half_cell)
            time_ok = abs(n_time - expected) <= 24
            if not (grid_ok and time_ok):
                log.warning(f"{path}: hours={n_time} (expected {expected}), "
                            f"x={float(ds.x.min())}-{float(ds.x.max())}, "
                            f"y={float(ds.y.min())}-{float(ds.y.max())} -> not usable")
                return False
            log.info(f"{path}: complete ({n_time} hours), variables: {sorted(ds.data_vars)}")
            return True
    except Exception as e:                    # unreadable / half-written file
        log.warning(f"{path}: could not be read ({e}) -> not usable")
        return False
 
 
def download_year(year):
    path = path_for(year)
    for attempt in range(1, ATTEMPTS_PER_YEAR + 1):
        try:
            log.info(f"--- {year}: starting download (attempt {attempt}) ---")
            cutout = atlite.Cutout(path, module="era5", x=X, y=Y,
                                   time=slice(f"{year}-01-01", end_date(year)))
            cutout.prepare(features=FEATURES)   # yearly requests (atlite default)
            if is_complete(path, year):
                log.info(f"--- {year}: done ---")
                return True
        except Exception as e:
            log.error(f"--- {year}: failed on attempt {attempt}: {e}")
        # move aside any half-finished file so the next attempt starts clean
        if os.path.exists(path):
            shutil.move(path, path + f".incomplete{attempt}")
    return False
 
 
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(os.path.join(OUT_DIR, "download_log.txt"))])
 
    # Reuse the 2024 test download if it is good
    if os.path.exists(TEST_FILE) and not os.path.exists(path_for(2024)):
        if is_complete(TEST_FILE, 2024):
            shutil.move(TEST_FILE, path_for(2024))
            log.info(f"Reusing {TEST_FILE} as {path_for(2024)} - 2024 will not be downloaded again")
        else:
            log.warning(f"{TEST_FILE} is incomplete or on a different grid - 2024 will be downloaded")
 
    done, failed = [], []
    for year in YEARS:
        if is_complete(path_for(year), year):
            log.info(f"{year}: already downloaded, skipping")
            done.append(year)
            continue
        if os.path.exists(path_for(year)):    # partial file from an interrupted run
            shutil.move(path_for(year), path_for(year) + ".incomplete")
        (done if download_year(year) else failed).append(year)
 
    log.info(f"Finished. Complete: {done}")
    if failed:
        log.warning(f"Failed: {failed} - run the script again to retry just these years")
 
 
if __name__ == "__main__":     
    main()