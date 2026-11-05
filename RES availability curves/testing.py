import atlite

cutout = atlite.Cutout("cutouts/moldova_era5_2015.nc")   # adjust the path to your folders
ws = cutout.data["wnd100m"].load()          
ws.sel(y=47.0, x=28.75, time="2015-07-15 14:00", method="nearest").compute().item()                    # 100 m wind speed, m/s

print(ws)                     # dimensions (time × y × x), coordinates, units
