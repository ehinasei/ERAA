import xarray as xr

ds = xr.open_dataset("cutouts/moldova_era5_2021.nc")
print(ds)                        # variables, dimensions, coordinates
print(ds.time.values[:5])        # first timestamps (UTC)

t = ds["temperature"]            # Kelvin, dims: time, y, x
print(t.attrs)                   # metadata stored with the variable

# one grid cell (nearest to Chișinău), first 6 hours, in K and °C
cell = t.sel(y=47.0, x=28.00, method="nearest")
print(cell.isel(time=slice(0, 6)).to_series())
print((cell.isel(time=slice(0, 6)) - 273.15).to_series())