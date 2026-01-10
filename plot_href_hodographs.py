import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
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
import traceback
import cfgrib
import glob 

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
REGION = [-83.5, -75.5, 32.5, 37.5]   
OUTPUT_DIR = "images"
GRID_SPACING = 25             
BOX_SIZE = 100000             
REQUESTED_LEVELS = [1000, 950, 925, 850, 700, 500, 400, 300, 250]

# --- SPC HREF STYLE CAPE CONFIGURATION ---
CAPE_LEVELS = [0, 100, 250, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 5000, 7000, 9000]
CAPE_COLORS = ['#ffffff', '#e1e1e1', '#c0c0c0', '#808080', '#626262', '#9dc2ff', '#4169e1', '#0000cd', '#00ff00', '#008000', '#ffff00', '#ff8c00', '#ff0000', '#ff00ff', '#800080']
CAPE_CMAP = mcolors.ListedColormap(CAPE_COLORS)
CAPE_NORM = mcolors.BoundaryNorm(CAPE_LEVELS, CAPE_CMAP.N)

# --- UH COLORS ---
UH_LEVELS = [25, 50, 75, 100, 150, 200, 250]
uh_colors = ['#c7f9cc', '#7cfc00', '#32cd32', '#008000', '#006400', '#000000']
UH_CMAP = mcolors.ListedColormap(uh_colors)
UH_NORM = mcolors.BoundaryNorm(UH_LEVELS, UH_CMAP.N)

def get_latest_run_time():
    now = datetime.datetime.utcnow()
    if now.hour >= 15: run = '12'; date = now
    elif now.hour >= 3: run = '00'; date = now
    else: run = '12'; date = now - datetime.timedelta(days=1)
    return date.strftime('%Y%m%d'), run, date

def download_file(date_str, run, fhr, prod_type):
    base_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/href/prod/href.{date_str}/ensprod"
    filename = f"href.t{run}z.conus.{prod_type}.f{fhr:02d}.grib2"
    url = f"{base_url}/{filename}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        with requests.get(url, stream=True, timeout=60, headers=headers) as r:
            if r.status_code == 404: return None
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        return filename
    except: return None

# --- FIXED HODOGRAPH PLOTTING ---
def plot_colored_hodograph(ax, u, v, levels):
    # 1. MetPy log-interpolation to get exact height points
    # This creates a 500m-resolution profile from 0 to 9km
    h_target = np.arange(0, 9001, 250) * units.m
    p_levels = levels * units.hPa
    h_levels = mpcalc.pressure_to_height_std(p_levels)
    
    # Sort everything by height increasing for interpolation
    sort_idx = np.argsort(h_levels)
    h_sorted = h_levels[sort_idx]
    u_sorted = u[sort_idx] * units.kts
    v_sorted = v[sort_idx] * units.kts

    u_interp = mpcalc.log_interp(h_target, h_sorted, u_sorted)
    v_interp = mpcalc.log_interp(h_target, h_sorted, v_sorted)

    # 2. Plot segments
    for k in range(len(h_target) - 1):
        z = h_target[k].to(units.km).m
        if z < 0.5: color = 'hotpink'
        elif 0.5 <= z < 3.0: color = 'red'
        elif 3.0 <= z < 6.0: color = 'green'
        elif 6.0 <= z <= 9.0: color = 'gold'
        else: continue
            
        ax.plot([u_interp[k].m, u_interp[k+1].m], [v_interp[k].m, v_interp[k+1].m], 
                color=color, linewidth=2.5)

def cleanup_old_runs(current_date, current_run):
    prefix = f"href_hodo_cape_{current_date}_{current_run}z"
    for f in glob.glob(os.path.join(OUTPUT_DIR, "href_hodo_cape_*.png")):
        if not os.path.basename(f).startswith(prefix):
            try: os.remove(f)
            except: pass

def process_forecast_hour(date_obj, date_str, run, fhr):
    mean_file = download_file(date_str, run, fhr, 'mean')
    pmmn_file = download_file(date_str, run, fhr, 'pmmn')
    if not mean_file: return

    try:
        ds_u = xr.open_dataset(mean_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'u'}})
        ds_v = xr.open_dataset(mean_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'v'}})
        ds_wind = xr.merge([ds_u, ds_v])
        ds_cape = xr.open_dataset(mean_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': 'cape', 'typeOfLevel': 'surface'}})
        
        ds_uh_max = None
        if pmmn_file:
            ds_pmmn_raw = xr.open_dataset(pmmn_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGroundLayer'}})
            ds_uh_max = ds_pmmn_raw[list(ds_pmmn_raw.data_vars)[0]]

        fig = plt.figure(figsize=(16, 12), facecolor='white')
        fig.subplots_adjust(bottom=0.18, top=0.93) 
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal())
        ax.set_extent(REGION)
        
        ax.add_feature(cfeature.COASTLINE, linewidth=2.0, zorder=10)
        ax.add_feature(cfeature.STATES, linewidth=1.5, edgecolor='black', zorder=10)

        if ds_cape is not None:
            cape_vals = np.nan_to_num(ds_cape['cape'].values, nan=0.0)
            cape_vals[cape_vals < 100] = 0
            cape_plot = ax.contourf(ds_cape.longitude, ds_cape.latitude, cape_vals, levels=CAPE_LEVELS, cmap=CAPE_CMAP, norm=CAPE_NORM, extend='max', alpha=0.6, transform=ccrs.PlateCarree())
            ax_cbar_cape = fig.add_axes([0.15, 0.10, 0.7, 0.02]) 
            cb_cape = plt.colorbar(cape_plot, cax=ax_cbar_cape, orientation='horizontal')
            cb_cape.set_label('Surface-based CAPE (J/kg)', fontsize=12, weight='bold')

        if ds_uh_max is not None:
            uh_masked = ds_uh_max.where(ds_uh_max >= 25)
            if np.nanmax(uh_masked.values) >= 25:
                max_plot = ax.contourf(ds_uh_max.longitude, ds_uh_max.latitude, uh_masked, levels=UH_LEVELS, cmap=UH_CMAP, norm=UH_NORM, extend='max', transform=ccrs.PlateCarree(), zorder=15)
                ax_cbar_max = fig.add_axes([0.3, 0.03, 0.4, 0.015]) 
                plt.colorbar(max_plot, cax=ax_cbar_max, orientation='horizontal', label='2-5km Max UH (>25 m$^2$/s$^2$)')

        legend_elements = [
            mlines.Line2D([], [], color='hotpink', lw=3, label='0-0.5 km'),
            mlines.Line2D([], [], color='red', lw=3, label='0.5-3 km'),
            mlines.Line2D([], [], color='green', lw=3, label='3-6 km'),
            mlines.Line2D([], [], color='gold', lw=3, label='6-9 km'),
            mlines.Line2D([], [], color='black', lw=0.5, alpha=0.5, label='Rings: 20 kts') 
        ]
        ax.legend(handles=legend_elements, loc='upper left', title="Hodograph Layers", framealpha=0.9).set_zorder(100)

        file_levels = ds_wind.isobaricInhPa.values
        current_levels = sorted([l for l in REQUESTED_LEVELS if l in file_levels], reverse=True)
        ds_wind_sub = ds_wind.sel(isobaricInhPa=current_levels)
        u_kts = ds_wind_sub['u'].metpy.convert_units('kts').values
        v_kts = ds_wind_sub['v'].metpy.convert_units('kts').values
        lons, lats = ds_wind_sub.longitude.values, ds_wind_sub.latitude.values

        for i in range(0, lons.shape[0], GRID_SPACING):
            for j in range(0, lons.shape[1], GRID_SPACING):
                if np.isnan(u_kts[:, i, j]).any(): continue
                lon_val = lons[i, j] - 360 if lons[i, j] > 180 else lons[i, j]
                if not (REGION[0] < lon_val < REGION[1] and REGION[2] < lats[i, j] < REGION[3]): continue
                proj_pnt = ax.projection.transform_point(lons[i, j], lats[i, j], ccrs.PlateCarree())
                sub_ax = ax.inset_axes([proj_pnt[0]-BOX_SIZE/2, proj_pnt[1]-BOX_SIZE/2, BOX_SIZE, BOX_SIZE], transform=ax.transData, zorder=20)
                h = Hodograph(sub_ax, component_range=80)
                h.add_grid(increment=20, color='black', alpha=0.3, linewidth=0.5)
                plot_colored_hodograph(h.ax, u_kts[:, i, j], v_kts[:, i, j], current_levels)
                sub_ax.axis('off')

        valid_time = date_obj + timedelta(hours=fhr)
        plt.suptitle(f"HREF Mean CAPE + PMMN UH Tracks | Valid: {valid_time.strftime('%a %H:%MZ')} (f{fhr:02d})", fontsize=20, weight='bold', y=0.98)
        plt.savefig(f"{OUTPUT_DIR}/href_hodo_cape_{date_str}_{run}z_f{fhr:02d}.png", bbox_inches='tight', dpi=100) 
        plt.close(fig)

    except: traceback.print_exc()
    finally:
        for f in [mean_file, pmmn_file]: 
            if f and os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    date_str, run, date_obj = get_latest_run_time()
    run_dt = datetime.datetime.strptime(f"{date_str} {run}", "%Y%m%d %H")
    for fhr in range(1, 49): process_forecast_hour(run_dt, date_str, run, fhr)
    cleanup_old_runs(date_str, run)
