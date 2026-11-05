import atlite


cutout = atlite.Cutout(r"C:\Users\ehina\Python\cutouts\moldova_era5_2021.nc")
ws = cutout.data["wnd100m"].load()

# 20:00 local (UTC+2) = 18:00 UTC; use 20:00 if you were reading time_utc
print(float(ws.sel(y=48.50, x=26.750, time="2021-01-01 18:00", method="nearest")))