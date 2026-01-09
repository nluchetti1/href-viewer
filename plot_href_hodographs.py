import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from metpy.plots import Hodograph
from metpy.units import units
import numpy as np
import datetime
from datetime import timedelta
import requests
import os
import sys
import warnings
import traceback
import cfgrib
import glob 

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
REGION = [-83.5, -75.5, 32.5, 37.5]   
OUTPUT_DIR = "images"
GRID_SPACING = 25              
BOX_SIZE = 100000              
REQUESTED_LEVELS = [1000, 925, 850, 700, 500, 250]

# --- CAPE COLORS & LEVELS ---
# Define levels such that 0-50 is the first bin
CAPE_LEVELS = [0, 50, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 
               2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 5000]

CAPE_COLORS = [
    '#ffffff', # 0-50: White
    '#d3d3d3', '#a9a9a9', '#808080', # Grays
    '#9dc2ff', '#70a1ff', '#00bfff', # Blues
    '#7fffd4', '#98fb98', '#adff2f', # Cyans/Greens
    '#ffff00', '#ffdb58', '#f4a460', # Yellows/Golds
    '#ff7f50', '#ff4500', '#cd5c5c', # Oranges/Reds
    '#a52a2a', '#ba55d3', '#9400d3', '#ff1493'  # Browns/Purples/Pinks
]

CAPE_CMAP = mcolors.ListedColormap(CAPE_COLORS)
CAPE_NORM = mcolors.BoundaryNorm(CAPE_LEVELS, CAPE_CMAP.N)

# --- UH COLORS ---
UH_LEVELS = [25, 50, 75, 100, 150, 200, 250]
uh_colors = [
    (0.6, 1, 0.6, 0.5), (0.3, 0.9, 0.3, 0.6), (0, 0.7, 0, 0.7),
    (0, 0.5, 0, 0.8), (0, 0.3, 0, 0.9), (0, 0, 0, 1.0)
]
UH_CMAP = mcolors.ListedColormap(uh_colors)
UH_NORM = mcolors.BoundaryNorm(UH_LEVELS, UH_CMAP.N)

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

def get_segment_color(p_start, p_end):
    avg_p = (p_start + p_end) / 2.0
    if avg_p >= 850: return 'magenta'
    elif 700 <= avg_p < 850: return 'red'
    elif 500 <= avg_p < 700: return 'green'
    else: return 'gold'

def plot_colored_hodograph(ax, u, v, levels):
    for k in range(len(u) - 1):
        color = get_segment_color(levels[k], levels[k+1])
        ax.plot([u[k], u[k+1]], [v[k], v[k+1]], color=color, linewidth=3.0)

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
        # Load Datasets
        ds_u = xr.open_dataset(mean_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'u'}})
        ds_v = xr.open_dataset(mean_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'v'}})
        ds_wind = xr.merge([ds_u, ds_v])
        ds_cape = xr.open_dataset(mean_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': 'cape', 'typeOfLevel': 'surface'}})
        
        ds_uh_max = None
        if pmmn_file:
            ds_pmmn_raw = xr.open_dataset(pmmn_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGroundLayer'}})
            ds_uh_max = ds_pmmn_raw[list(ds_pmmn_raw.data_vars)[0]]

        # Setup Figure
        fig = plt.figure(figsize=(16, 12), facecolor='white')
        fig.subplots_adjust(bottom=0.18, top=0.93)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal())
        ax.set_extent(REGION)
        
        ax.add_feature(cfeature.COASTLINE, linewidth=2.0, zorder=10)
        ax.add_feature(cfeature.STATES, linewidth=1.5, edgecolor='black', zorder=10)

        # --- PLOT CAPE ---
        if ds_cape is not None:
            cape_vals = np.nan_to_num(ds_cape['cape'].values, nan=0.0)
            cape_vals[cape_vals < 0] = 0
            
            cape_plot = ax.contourf(ds_cape.longitude, ds_cape.latitude, cape_vals, 
                                    levels=CAPE_LEVELS, cmap=CAPE_CMAP, norm=CAPE_NORM,
                                    extend='max', alpha=0.6, transform=ccrs.PlateCarree())
            
            # Colorbar Positioning
            ax_cbar_cape = fig.add_axes([0.15, 0.10, 0.7, 0.02]) 
            cb_cape = plt.colorbar(cape_plot, cax=ax_cbar_cape, orientation='horizontal')
            
            # Fixed Ticks to ensure labels don't bunch up or align to centers
            tick_locs = [0, 750, 1500, 2250, 3000, 3750, 4500]
            cb_cape.set_ticks(tick_locs)
            cb_cape.ax.set_xticklabels([str(t) for t in tick_locs], fontsize=10)
            cb_cape.set_label('SBCAPE (J/kg)', fontsize=12, weight='bold')

        # --- PLOT UH ---
        if ds_uh_max is not None:
            uh_masked = ds_uh_max.where(ds_uh_max >= 25)
            if np.nanmax(uh_masked.values) >= 25:
                max_plot = ax.contourf(ds_uh_max.longitude, ds_uh_max.latitude, uh_masked,
                                       levels=UH_LEVELS, cmap=UH_CMAP, norm=UH_NORM,
                                       extend='max', transform=ccrs.PlateCarree(), zorder=15)
                ax_cbar_max = fig.add_axes([0.3, 0.03, 0.4, 0.015]) 
                plt.colorbar(max_plot, cax=ax_cbar_max, orientation='horizontal', label='2-5km Max UH (>25 m$^2$/s$^2$)')

        # Hodographs
        u_kts = ds_wind['u'].metpy.convert_units('kts').values
        v_kts = ds_wind['v'].metpy.convert_units('kts').values
        lons, lats = ds_wind.longitude.values, ds_wind.latitude.values
        
        for i in range(0, lons.shape[0], GRID_SPACING):
            for j in range(0, lons.shape[1], GRID_SPACING):
                if np.isnan(u_kts[:, i, j]).any(): continue
                lon_val = lons[i, j] - 360 if lons[i, j] > 180 else lons[i, j]
                if not (REGION[0] < lon_val < REGION[1] and REGION[2] < lats[i, j] < REGION[3]): continue
                
                proj_pnt = ax.projection.transform_point(lons[i, j], lats[i, j], ccrs.PlateCarree())
                sub_ax = ax.inset_axes([proj_pnt[0]-BOX_SIZE/2, proj_pnt[1]-BOX_SIZE/2, BOX_SIZE, BOX_SIZE], transform=ax.transData, zorder=20)
                h = Hodograph(sub_ax, component_range=80)
                h.add_grid(increment=20, color='black', alpha=0.3, linewidth=0.5)
                plot_colored_hodograph(h.ax, u_kts[:, i, j], v_kts[:, i, j], REQUESTED_LEVELS)
                sub_ax.axis('off')

        # Legend & Titles
        legend_elements = [
            mlines.Line2D([], [], color='magenta', lw=3, label='0-1.5 km'),
            mlines.Line2D([], [], color='red', lw=3, label='1.5-3 km'),
            mlines.Line2D([], [], color='green', lw=3, label='3-6 km'),
            mlines.Line2D([], [], color='gold', lw=3, label='6-9 km'),
            mlines.Line2D([], [], color='black', lw=0.5, alpha=0.5, label='Rings: 20 kts') 
        ]
        ax.legend(handles=legend_elements, loc='upper left', title="Hodograph Layers", framealpha=0.9).set_zorder(100)

        valid_time = date_obj + timedelta(hours=fhr)
        plt.suptitle(f"HREF Mean CAPE + PMMN UH Tracks | Run: {date_str} {run}Z | Valid: {valid_time.strftime('%a %H:%MZ')} (f{fhr:02d})", 
                     fontsize=20, weight='bold', y=0.98)
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
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
