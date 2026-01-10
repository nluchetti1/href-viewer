import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from metpy.plots import Hodograph
from metpy.units import units
import metpy.calc as mpcalc
import numpy as np
import datetime
from datetime import timedelta
import requests
import os
import warnings
import traceback

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
REGION = [-83.5, -75.5, 32.5, 37.5]   
OUTPUT_DIR = "images"
GRID_SPACING = 25             
BOX_SIZE = 100000             

# Expanded levels to capture the new 0.5km and 9km requirements
REQUESTED_LEVELS = [1000, 950, 925, 850, 700, 500, 400, 300, 250]

# --- CAPE SETTINGS ---
CAPE_LEVELS = np.arange(0, 5001, 250) 
CAPE_COLORS = [
    '#ffffff', '#e0e0e0', '#b0b0b0', '#808080', '#6495ed', 
    '#4169e1', '#00bfff', '#40e0d0', '#adff2f', '#ffff00', 
    '#ffda00', '#ffa500', '#ff8c00', '#ff4500', '#ff0000', 
    '#b22222', '#8b0000', '#800080', '#9400d3', '#ff1493'
]
CAPE_CMAP = mcolors.ListedColormap(CAPE_COLORS)

# --- Precise Pressure Thresholds using MetPy ---
P05 = mpcalc.height_to_pressure_std(0.5 * units.km).m
P3  = mpcalc.height_to_pressure_std(3 * units.km).m
P6  = mpcalc.height_to_pressure_std(6 * units.km).m
P9  = mpcalc.height_to_pressure_std(9 * units.km).m

def get_latest_run_time():
    now = datetime.datetime.utcnow()
    if now.hour >= 15:
        run = '12'
        date = now
    elif now.hour >= 3:
        run = '00'
        date = now
    else:
        run = '12'
        date = now - datetime.timedelta(days=1)
    return date.strftime('%Y%m%d'), run, date

def download_href_mean(date_str, run, fhr):
    base_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/href/prod/href.{date_str}/ensprod"
    filename = f"href.t{run}z.conus.mean.f{fhr:02d}.grib2"
    url = f"{base_url}/{filename}"
    print(f"\n[f{fhr:02d}] Downloading: {url}")
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        with requests.get(url, stream=True, timeout=60, headers=headers) as r:
            if r.status_code == 404: return None
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        return filename
    except Exception: return None

def get_segment_color(p_start, p_end):
    avg_p = (p_start + p_end) / 2.0
    if avg_p >= P05:
        return 'hotpink'  # 0 to 0.5 km
    elif P3 <= avg_p < P05:
        return 'red'      # 0.5 to 3 km
    elif P6 <= avg_p < P3:
        return 'green'    # 3 to 6 km
    elif P9 <= avg_p < P6:
        return 'gold'     # 6 to 9 km
    else:
        return 'cyan'     # Above 9 km

def plot_colored_hodograph(ax, u, v, levels):
    for k in range(len(u) - 1):
        color = get_segment_color(levels[k], levels[k+1])
        ax.plot([u[k], u[k+1]], [v[k], v[k+1]], color=color, linewidth=2.5)

def process_forecast_hour(date_obj, date_str, run, fhr):
    grib_file = download_href_mean(date_str, run, fhr)
    if not grib_file: return

    try:
        ds_u = xr.open_dataset(grib_file, engine='cfgrib', filter_by_keys={'typeOfLevel': 'isobaricInhPa', 'shortName': 'u'})
        ds_v = xr.open_dataset(grib_file, engine='cfgrib', filter_by_keys={'typeOfLevel': 'isobaricInhPa', 'shortName': 'v'})
        ds_wind = xr.merge([ds_u, ds_v])
        
        try:
            ds_cape = xr.open_dataset(grib_file, engine='cfgrib', filter_by_keys={'shortName': 'cape', 'typeOfLevel': 'surface'})
        except:
            ds_cape = None

        file_levels = ds_wind.isobaricInhPa.values
        available_levels = sorted([l for l in REQUESTED_LEVELS if l in file_levels], reverse=True)
        
        if len(available_levels) < 3: return
        
        ds_wind = ds_wind.sel(isobaricInhPa=available_levels)
        u = ds_wind['u'].metpy.convert_units('kts').values
        v = ds_wind['v'].metpy.convert_units('kts').values
        lons, lats = ds_wind.longitude.values, ds_wind.latitude.values

        fig = plt.figure(figsize=(18, 12))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal())
        ax.set_extent(REGION)
        ax.add_feature(cfeature.STATES, linewidth=1.0, edgecolor='black', zorder=10)

        if ds_cape is not None:
            cape_vals = np.nan_to_num(ds_cape['cape'].values, nan=0.0)
            cape_plot = ax.contourf(ds_cape.longitude, ds_cape.latitude, cape_vals, 
                                    levels=CAPE_LEVELS, cmap=CAPE_CMAP, extend='max', alpha=0.5, transform=ccrs.PlateCarree())
            plt.colorbar(cape_plot, ax=ax, orientation='horizontal', pad=0.02, aspect=50, shrink=0.8, label='SBCAPE (J/kg)')

        # Updated Legend with precise height labels
        legend_elements = [
            mlines.Line2D([], [], color='hotpink', lw=3, label='0-0.5 km'),
            mlines.Line2D([], [], color='red', lw=3, label='0.5-3 km'),
            mlines.Line2D([], [], color='green', lw=3, label='3-6 km'),
            mlines.Line2D([], [], color='gold', lw=3, label='6-9 km'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', title="Shear Layers", framealpha=0.9).set_zorder(100)

        for i in range(0, lons.shape[0], GRID_SPACING):
            for j in range(0, lons.shape[1], GRID_SPACING):
                if np.isnan(u[:, i, j]).any(): continue
                
                lon, lat = lons[i, j], lats[i, j]
                check_lon = lon - 360 if lon > 180 else lon
                if not (REGION[0] < check_lon < REGION[1] and REGION[2] < lat < REGION[3]): continue

                proj_pnt = ax.projection.transform_point(lon, lat, ccrs.PlateCarree())
                bounds = [proj_pnt[0] - BOX_SIZE/2, proj_pnt[1] - BOX_SIZE/2, BOX_SIZE, BOX_SIZE]
                sub_ax = ax.inset_axes(bounds, transform=ax.transData, zorder=20)
                
                h = Hodograph(sub_ax, component_range=80)
                h.add_grid(increment=20, color='black', alpha=0.2, linewidth=0.5)
                plot_colored_hodograph(h.ax, u[:, i, j], v[:, i, j], available_levels)
                sub_ax.axis('off')

        valid_str = (date_obj + timedelta(hours=fhr)).strftime("%a %H:%MZ")
        plt.title(f"HREF Mean CAPE & Hodographs | Valid: {valid_str} (f{fhr:02d})", fontsize=16, weight='bold')
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(f"{OUTPUT_DIR}/hodo_f{fhr:02d}.png", bbox_inches='tight', dpi=100)
        plt.close(fig)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if os.path.exists(grib_file): os.remove(grib_file)

if __name__ == "__main__":
    date_str, run, date_obj = get_latest_run_time()
    run_dt = datetime.datetime.strptime(f"{date_str} {run}", "%Y%m%d %H")
    for fhr in range(1, 37):
        process_forecast_hour(run_dt, date_str, run, fhr)
