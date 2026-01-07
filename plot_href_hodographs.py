import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
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

# --- CAPE COLORS (Background) ---
CAPE_LEVELS = np.arange(0, 5001, 250) 
CAPE_COLORS = [
    '#ffffff', '#f5f5f5', '#b0b0b0', '#808080', 
    '#6495ed', '#4169e1', '#00bfff', '#40e0d0', 
    '#adff2f', '#ffff00', '#ffda00', '#ffa500', 
    '#ff8c00', '#ff4500', '#ff0000', '#b22222', 
    '#8b0000', '#800080', '#9400d3', '#ff1493'
]
CAPE_CMAP = mcolors.ListedColormap(CAPE_COLORS)

# --- UH COLORS (Green Scale - Positive Only) ---
UH_LEVELS = [25, 50, 75, 100, 150, 200, 250]
uh_colors = [
    (0.6, 1, 0.6, 0.5),   # 25-50:  Pale Green
    (0.3, 0.9, 0.3, 0.6), # 50-75:  Light Green
    (0, 0.7, 0, 0.7),     # 75-100: Medium Green
    (0, 0.5, 0, 0.8),     # 100-150: Dark Green
    (0, 0.3, 0, 0.9),     # 150-200: Forest Green
    (0, 0, 0, 1.0)        # 200+:   Black
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
    
    if prod_type == 'mean':
        print(f"\n[f{fhr:02d}] Downloading Mean & PMMN data...")
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    try:
        with requests.get(url, stream=True, timeout=60, headers=headers) as r:
            if r.status_code == 404:
                return None
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return filename
    except Exception:
        return None

def get_segment_color(pressure_start, pressure_end):
    avg_p = (pressure_start + pressure_end) / 2.0
    if avg_p >= 850: return 'magenta'
    elif 700 <= avg_p < 850: return 'red'
    elif 500 <= avg_p < 700: return 'green'
    else: return 'gold'

def plot_colored_hodograph(ax, u, v, levels):
    for k in range(len(u) - 1):
        p_start = levels[k]
        p_end = levels[k+1]
        color = get_segment_color(p_start, p_end)
        ax.plot([u[k], u[k+1]], [v[k], v[k+1]], color=color, linewidth=3.0)

def cleanup_old_runs(current_date, current_run):
    print("\n" + "="*40)
    print("CLEANUP: Removing old run data...")
    current_prefix = f"href_hodo_cape_{current_date}_{current_run}z"
    all_files = glob.glob(os.path.join(OUTPUT_DIR, "href_hodo_cape_*.png"))
    
    removed_count = 0
    for filepath in all_files:
        filename = os.path.basename(filepath)
        if not filename.startswith(current_prefix):
            try:
                os.remove(filepath)
                removed_count += 1
            except Exception:
                pass
                
    print(f"Cleanup Complete. Removed {removed_count} old files.")
    print("="*40 + "\n")

def process_forecast_hour(date_obj, date_str, run, fhr):
    mean_file = download_file(date_str, run, fhr, 'mean')
    pmmn_file = download_file(date_str, run, fhr, 'pmmn')
    
    if not mean_file:
        print(f"Skipping f{fhr:02d} (Mean file missing)")
        return

    try:
        print(f"       Loading Data...")
        
        # --- LOAD MEAN DATA ---
        ds_u = xr.open_dataset(mean_file, engine='cfgrib', 
                               backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'u'}})
        ds_v = xr.open_dataset(mean_file, engine='cfgrib', 
                               backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'v'}})
        ds_wind = xr.merge([ds_u, ds_v])
        
        ds_cape = None
        try:
            ds_cape = xr.open_dataset(mean_file, engine='cfgrib', 
                                      backend_kwargs={'filter_by_keys': {'shortName': 'cape', 'typeOfLevel': 'surface'}})
        except: pass

        # --- LOAD PMMN DATA ---
        ds_uh_max = None
        if pmmn_file:
            try:
                ds_pmmn_raw = xr.open_dataset(pmmn_file, engine='cfgrib', 
                                              backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGroundLayer'}})
                var_name = list(ds_pmmn_raw.data_vars)[0]
                ds_uh_max = ds_pmmn_raw[var_name]
                print(f"       Found PMMN UH (Var: {var_name}). Max: {ds_uh_max.values.max():.1f}")
            except Exception as e:
                print(f"       PMMN UH Load Failed: {e}")

        # --- PLOTTING ---
        file_levels = ds_wind.isobaricInhPa.values
        available_levels = sorted([l for l in REQUESTED_LEVELS if l in file_levels], reverse=True)
        if len(available_levels) < 3: return
        
        ds_wind = ds_wind.sel(isobaricInhPa=available_levels)
        u = ds_wind['u'].metpy.convert_units('kts')
        v = ds_wind['v'].metpy.convert_units('kts')
        level_values = available_levels

        print(f"       Generating Map...")
        
        # 1. FORCE WHITE FIGURE BACKGROUND
        fig = plt.figure(figsize=(16, 12), facecolor='white') 
        fig.subplots_adjust(bottom=0.18, top=0.93)

        ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal())
        ax.set_extent(REGION)
        
        # 2. FORCE WHITE AXES BACKGROUND
        ax.set_facecolor('white')
        
        # 3. FORCE WHITE FEATURES (To cover any gray)
        ax.add_feature(cfeature.LAND, facecolor='white', zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
        
        ax.add_feature(cfeature.COASTLINE, linewidth=2.0, zorder=10)
        ax.add_feature(cfeature.BORDERS, linewidth=2.0, zorder=10)
        ax.add_feature(cfeature.STATES, linewidth=1.5, edgecolor='black', zorder=10)

        # Plot CAPE
        if ds_cape is not None:
            cape_data = ds_cape['cape']
            # FIX: Convert NaNs to 0.0 immediately
            cape_vals = np.nan_to_num(cape_data.values, nan=0.0)
            
            # Only plot if there is actual CAPE > 10
            if np.max(cape_vals) > 10:
                try:
                    cape_plot = ax.contourf(cape_data.longitude, cape_data.latitude, cape_vals, 
                                            levels=CAPE_LEVELS, cmap=CAPE_CMAP, 
                                            extend='max', alpha=0.6, transform=ccrs.PlateCarree())
                    
                    ax_cbar_cape = fig.add_axes([0.15, 0.10, 0.7, 0.02]) 
                    plt.colorbar(cape_plot, cax=ax_cbar_cape, orientation='horizontal', label='SBCAPE (J/kg)')
                except Exception as e:
                    print(f"       [!] Plotting CAPE failed: {e}")
            else:
                print("       [!] CAPE data empty/NaN (Max < 10). Map will be white.")

        # Plot MAX UH
        if ds_uh_max is not None:
            uh_vals = ds_uh_max.values
            if np.nanmax(uh_vals) > 20:
                print("       Plotting Max UH Swaths (Green)...")
                try:
                    max_plot = ax.contourf(ds_uh_max.longitude, ds_uh_max.latitude, uh_vals,
                                          levels=UH_LEVELS, cmap=UH_CMAP, norm=UH_NORM,
                                          extend='max', transform=ccrs.PlateCarree(), zorder=15)
                    
                    ax_cbar_max = fig.add_axes([0.3, 0.03, 0.4, 0.015]) 
                    plt.colorbar(max_plot, cax=ax_cbar_max, orientation='horizontal', label='2-5km Max UH (>25 m$^2$/s$^2$)')
                except: pass

        # Legend
        legend_elements = [
            mlines.Line2D([], [], color='magenta', lw=3, label='0-1.5 km'),
            mlines.Line2D([], [], color='red', lw=3, label='1.5-3 km'),
            mlines.Line2D([], [], color='green', lw=3, label='3-6 km'),
            mlines.Line2D([], [], color='gold', lw=3, label='6-9 km'),
            mlines.Line2D([], [], color='black', lw=0.5, alpha=0.5, label='Rings: 20 kts') 
        ]
        ax.legend(handles=legend_elements, loc='upper left', title="Hodograph Layers", 
                  framealpha=0.9, fontsize=11, title_fontsize=12).set_zorder(100)

        # Hodographs
        lons = u.longitude.values
        lats = u.latitude.values
        u_data = u.values
        v_data = v.values

        for i in range(0, lons.shape[0], GRID_SPACING):
            for j in range(0, lons.shape[1], GRID_SPACING):
                if np.isnan(u_data[:, i, j]).any(): continue
                curr_lon = lons[i, j]
                curr_lat = lats[i, j]
                check_lon = curr_lon - 360 if curr_lon > 180 else curr_lon
                if not (REGION[0] < check_lon < REGION[1] and REGION[2] < curr_lat < REGION[3]): continue
                try:
                    proj_pnt = ax.projection.transform_point(curr_lon, curr_lat, ccrs.PlateCarree())
                    bounds = [proj_pnt[0] - BOX_SIZE/2, proj_pnt[1] - BOX_SIZE/2, BOX_SIZE, BOX_SIZE]
                    sub_ax = ax.inset_axes(bounds, transform=ax.transData, zorder=20)
                    h = Hodograph(sub_ax, component_range=80)
                    h.add_grid(increment=20, color='black', alpha=0.3, linewidth=0.5)
                    plot_colored_hodograph(h.ax, u_data[:, i, j], v_data[:, i, j], level_values)
                    sub_ax.axis('off')
                except: continue

        # Save
        valid_time = date_obj + timedelta(hours=fhr)
        valid_str = valid_time.strftime("%a %H:%MZ") 
        plt.suptitle(f"HREF Mean CAPE + PMMN UH Tracks | Run: {date_str} {run}Z | Valid: {valid_str} (f{fhr:02d})", 
                     fontsize=20, weight='bold', y=0.98)
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = f"{OUTPUT_DIR}/href_hodo_cape_{date_str}_{run}z_f{fhr:02d}.png"
        plt.savefig(out_path, bbox_inches='tight', dpi=90) 
        plt.close(fig)

    except Exception as e:
        print(f"Error processing f{fhr:02d}: {e}")
        traceback.print_exc()
    finally:
        if mean_file and os.path.exists(mean_file): os.remove(mean_file)
        if pmmn_file and os.path.exists(pmmn_file): os.remove(pmmn_file)

if __name__ == "__main__":
    date_str, run, date_obj = get_latest_run_time()
    run_dt = datetime.datetime.strptime(f"{date_str} {run}", "%Y%m%d %H")
    print(f"Starting HREF (Mean CAPE + PMMN UH) generation for {date_str} {run}Z")
    
    for fhr in range(1, 49):
        process_forecast_hour(run_dt, date_str, run, fhr)
    
    cleanup_old_runs(date_str, run)
