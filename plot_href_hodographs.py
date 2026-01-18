import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from metpy.plots import Hodograph
import numpy as np
import datetime
from datetime import timedelta
import requests
import os
import traceback
import glob
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
REGION = [-83.5, -75.5, 32.5, 37.5]    
OUTPUT_DIR = "images"
GRID_SPACING = 25              
BOX_SIZE = 100000              
REQUESTED_LEVELS = [1000, 925, 850, 700, 500, 250]

# --- SPC HREF STYLE CAPE CONFIGURATION (0-100 White) ---
CAPE_LEVELS = [0, 100, 250, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 5000, 7000, 9000]

CAPE_COLORS = [
    '#ffffff', # 0-100: White
    '#e1e1e1', '#c0c0c0', '#808080', '#626262', # Grays
    '#9dc2ff', '#4169e1', '#0000cd', # Blues
    '#00ff00', '#008000', # Greens
    '#ffff00', # Yellow
    '#ff8c00', # Orange
    '#ff0000', # Red
    '#ff00ff', # Magenta
    '#800080'  # Purple
]

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
    
    # --- FORCE FRESH DOWNLOAD ---
    # We remove the file if it exists to ensure we aren't using cached/corrupt data
    if os.path.exists(filename):
        try: os.remove(filename)
        except: pass
    # ----------------------------

    try:
        print(f"Downloading {filename}...")
        with requests.get(url, stream=True, timeout=60, headers=headers) as r:
            if r.status_code == 404: 
                print(f"Missing: {url}")
                return None
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        return filename
    except Exception as e:
        print(f"Download failed: {e}")
        return None

def get_segment_color(p_start, p_end):
    avg_p = (p_start + p_end) / 2.0
    if avg_p >= 850: return 'magenta'
    elif 700 <= avg_p < 850: return 'red'
    elif 500 <= avg_p < 700: return 'green'
    else: return 'gold'

def plot_colored_hodograph(ax, u, v, levels):
    safe_len = min(len(u), len(levels))
    for k in range(safe_len - 1):
        color = get_segment_color(levels[k], levels[k+1])
        ax.plot([u[k], u[k+1]], [v[k], v[k+1]], color=color, linewidth=2.5)

def cleanup_old_runs(current_date, current_run):
    prefix = f"href_hodo_cape_{current_date}_{current_run}z"
    for f in glob.glob(os.path.join(OUTPUT_DIR, "href_hodo_cape_*.png")):
        if not os.path.basename(f).startswith(prefix):
            try: os.remove(f)
            except: pass

def process_forecast_hour(date_obj, date_str, run, fhr):
    print(f"\nProcessing F{fhr:02d}...")
    mean_file = download_file(date_str, run, fhr, 'mean')
    pmmn_file = download_file(date_str, run, fhr, 'pmmn')
    
    if not mean_file: 
        print("Mean file unavailable.")
        return

    try:
        # Load datasets
        ds_u = xr.open_dataset(mean_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'u'}})
        ds_v = xr.open_dataset(mean_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'v'}})
        ds_wind = xr.merge([ds_u, ds_v])
        ds_cape = xr.open_dataset(mean_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': 'cape', 'typeOfLevel': 'surface'}})
        
        ds_uh_max = None
        if pmmn_file:
            ds_pmmn_raw = xr.open_dataset(pmmn_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGroundLayer'}})
            ds_uh_max = ds_pmmn_raw[list(ds_pmmn_raw.data_vars)[0]]

        # --- DEBUGGING OUTPUT ---
        cape_min = np.nanmin(ds_cape['cape'].values)
        cape_max = np.nanmax(ds_cape['cape'].values)
        print(f"   [DEBUG] CAPE Range: {cape_min:.1f} to {cape_max:.1f} J/kg")
        # ------------------------

        fig = plt.figure(figsize=(16, 12), facecolor='white')
        fig.subplots_adjust(bottom=0.18, top=0.93)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal())
        ax.set_extent(REGION)
        
        ax.add_feature(cfeature.COASTLINE, linewidth=2.0, zorder=10)
        ax.add_feature(cfeature.STATES, linewidth=1.5, edgecolor='black', zorder=10)

        # --- PLOT CAPE (PCOLORMESH FIX) ---
        if ds_cape is not None:
            cape_vals = np.nan_to_num(ds_cape['cape'].values.squeeze(), nan=0.0)
            cape_vals[cape_vals < 100] = 0
            
            lats = ds_cape.latitude.values
            lons = ds_cape.longitude.values
            lons = (lons + 180) % 360 - 180 

            # SWITCH TO PCOLORMESH
            # pcolormesh draws individual grid cells rather than wrapping polygons.
            # This completely eliminates "blanket" wrapping artifacts.
            cape_plot = ax.pcolormesh(lons, lats, cape_vals, 
                                      cmap=CAPE_CMAP, norm=CAPE_NORM,
                                      shading='auto', transform=ccrs.PlateCarree(), alpha=0.6)
            
            ax_cbar_cape = fig.add_axes([0.15, 0.10, 0.7, 0.02]) 
            cb_cape = plt.colorbar(cape_plot, cax=ax_cbar_cape, orientation='horizontal')
            
            spc_ticks = [100, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 7000, 9000]
            cb_cape.set_ticks(spc_ticks)
            cb_cape.ax.set_xticklabels([str(t) for t in spc_ticks], fontsize=10)
            cb_cape.set_label('Surface-based CAPE (J/kg)', fontsize=12, weight='bold')
            
        # --- PLOT UH (PCOLORMESH FIX) ---
        if ds_uh_max is not None:
            uh_vals = ds_uh_max.values.squeeze()
            uh_masked = np.where(uh_vals >= 25, uh_vals, np.nan)
            
            uh_lats = ds_uh_max.latitude.values
            uh_lons = ds_uh_max.longitude.values
            uh_lons = (uh_lons + 180) % 360 - 180

            if np.all(np.isnan(uh_masked)):
                has_data = False
            else:
                has_data = np.nanmax(uh_masked) >= 25

            if has_data:
                # Use pcolormesh for UH as well
                cf_uh = ax.pcolormesh(uh_lons, uh_lats, uh_masked,
                                      cmap=UH_CMAP, norm=UH_NORM,
                                      shading='auto', transform=ccrs.PlateCarree(), zorder=15)
                mappable = cf_uh
            else:
                mappable = plt.cm.ScalarMappable(norm=UH_NORM, cmap=UH_CMAP)
                mappable.set_array([]) 
            
            ax_cbar_max = fig.add_axes([0.3, 0.03, 0.4, 0.015]) 
            plt.colorbar(mappable, cax=ax_cbar_max, orientation='horizontal', 
                         label='2-5km Max UH (>25 m$^2$/s$^2$)', extend='max')

        # --- HODOGRAPHS ---
        legend_elements = [
            mlines.Line2D([], [], color='magenta', lw=3, label='0-1.5 km'),
            mlines.Line2D([], [], color='red', lw=3, label='1.5-3 km'),
            mlines.Line2D([], [], color='green', lw=3, label='3-6 km'),
            mlines.Line2D([], [], color='gold', lw=3, label='6-9 km'),
            mlines.Line2D([], [], color='black', lw=0.5, alpha=0.5, label='Rings: 20 kts') 
        ]
        ax.legend(handles=legend_elements, loc='upper left', title="Hodograph Layers", framealpha=0.9).set_zorder(100)

        u_kts = ds_wind['u'].metpy.convert_units('kts').values.squeeze()
        v_kts = ds_wind['v'].metpy.convert_units('kts').values.squeeze()
        
        lons_wind = ds_wind.longitude.values
        lats_wind = ds_wind.latitude.values
        
        for i in range(0, lons_wind.shape[0], GRID_SPACING):
            for j in range(0, lons_wind.shape[1], GRID_SPACING):
                
                # Check for NaNs across the vertical profile
                if np.isnan(u_kts[:, i, j]).any(): continue
                
                lon_val = lons_wind[i, j]
                lon_val = lon_val - 360 if lon_val > 180 else lon_val
                
                if not (REGION[0] < lon_val < REGION[1] and REGION[2] < lats_wind[i, j] < REGION[3]): continue
                
                proj_pnt = ax.projection.transform_point(lons_wind[i, j], lats_wind[i, j], ccrs.PlateCarree())
                sub_ax = ax.inset_axes([proj_pnt[0]-BOX_SIZE/2, proj_pnt[1]-BOX_SIZE/2, BOX_SIZE, BOX_SIZE], transform=ax.transData, zorder=20)
                h = Hodograph(sub_ax, component_range=80)
                h.add_grid(increment=20, color='black', alpha=0.3, linewidth=0.5)
                
                plot_colored_hodograph(h.ax, u_kts[:, i, j], v_kts[:, i, j], REQUESTED_LEVELS)
                sub_ax.axis('off')

        valid_time = date_obj + timedelta(hours=fhr)
        valid_str = valid_time.strftime("%a %H:%MZ") 
        plt.suptitle(f"HREF Mean CAPE + PMMN UH Tracks | Run: {date_str} {run}Z | Valid: {valid_str} (f{fhr:02d})", 
                     fontsize=20, weight='bold', y=0.98)
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filename_png = f"{OUTPUT_DIR}/href_hodo_cape_{date_str}_{run}z_f{fhr:02d}.png"
        plt.savefig(filename_png, bbox_inches='tight', dpi=100) 
        print(f"   Saved: {filename_png}")
        plt.close(fig)

    except Exception: 
        print(f"Error processing F{fhr}:")
        traceback.print_exc()
    finally:
        for f in [mean_file, pmmn_file]: 
            if f and os.path.exists(f): 
                try: os.remove(f)
                except: pass

if __name__ == "__main__":
    date_str, run, date_obj = get_latest_run_time()
    print(f"Starting Run: {date_str} {run}Z")
    run_dt = datetime.datetime.strptime(f"{date_str} {run}", "%Y%m%d %H")
    
    for fhr in range(1, 49): 
        process_forecast_hour(run_dt, date_str, run, fhr)
    
    cleanup_old_runs(date_str, run)
    print("Done.")
