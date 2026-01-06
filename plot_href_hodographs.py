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

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
# ZOOMED DOMAIN: Tight focus on SC/NC/VA
REGION = [-83.5, -75.5, 32.5, 37.5]   
OUTPUT_DIR = "images"

# --- TUNING SETTINGS ---
GRID_SPACING = 25             
BOX_SIZE = 100000              

# Preferred Levels (Surface -> Aloft)
REQUESTED_LEVELS = [1000, 925, 850, 700, 500, 250]

# --- CAPE SETTINGS (Pale Gray Fix) ---
CAPE_LEVELS = np.arange(0, 5001, 250) 
CAPE_COLORS = [
    '#ffffff', '#f5f5f5', '#b0b0b0', '#808080', 
    '#6495ed', '#4169e1', '#00bfff', '#40e0d0', 
    '#adff2f', '#ffff00', '#ffda00', '#ffa500', 
    '#ff8c00', '#ff4500', '#ff0000', '#b22222', 
    '#8b0000', '#800080', '#9400d3', '#ff1493'
]
CAPE_CMAP = mcolors.ListedColormap(CAPE_COLORS)

# --- UH SETTINGS (0-3 km Rotation) ---
# Since we are plotting the ENSEMBLE MEAN, values are smoothed out. 
# We use lower thresholds than you would for a single HRRR run.
# Positive (Cyclonic)
UH_MAX_LEVELS = [20, 40, 60, 80, 100] 
UH_MAX_COLORS = ['gold', 'orange', 'orangered', 'red', 'darkred']

# Negative (Anti-Cyclonic)
UH_MIN_LEVELS = [-100, -80, -60, -40, -20]
UH_MIN_COLORS = ['navy', 'mediumblue', 'blue', 'dodgerblue', 'cyan']

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
    """Downloads the HREF Mean file for a specific forecast hour."""
    base_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/href/prod/href.{date_str}/ensprod"
    filename = f"href.t{run}z.conus.mean.f{fhr:02d}.grib2"
    url = f"{base_url}/{filename}"
    
    print(f"\n[f{fhr:02d}] Downloading: {url}")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    try:
        with requests.get(url, stream=True, timeout=60, headers=headers) as r:
            if r.status_code == 404:
                print(f"File not found (404). Forecast f{fhr:02d} may not be ready yet.")
                return None
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return filename
    except Exception as e:
        print(f"Download failed: {e}")
        return None

def get_segment_color(pressure_start, pressure_end):
    """Determines color based on the pressure level of the segment."""
    avg_p = (pressure_start + pressure_end) / 2.0
    if avg_p >= 850: return 'magenta'
    elif 700 <= avg_p < 850: return 'red'
    elif 500 <= avg_p < 700: return 'green'
    else: return 'gold'

def plot_colored_hodograph(ax, u, v, levels):
    """Plots a multi-colored hodograph."""
    for k in range(len(u) - 1):
        p_start = levels[k]
        p_end = levels[k+1]
        color = get_segment_color(p_start, p_end)
        ax.plot([u[k], u[k+1]], [v[k], v[k+1]], color=color, linewidth=3.0)

def process_forecast_hour(date_obj, date_str, run, fhr):
    grib_file = download_href_mean(date_str, run, fhr)
    if not grib_file:
        return

    try:
        print(f"[f{fhr:02d}] Loading GRIB data...")
        
        # --- 1. Load Winds ---
        ds_u = xr.open_dataset(grib_file, engine='cfgrib', 
                               filter_by_keys={'typeOfLevel': 'isobaricInhPa', 'shortName': 'u'})
        ds_v = xr.open_dataset(grib_file, engine='cfgrib', 
                               filter_by_keys={'typeOfLevel': 'isobaricInhPa', 'shortName': 'v'})
        ds_wind = xr.merge([ds_u, ds_v])
        
        # --- 2. Load CAPE ---
        ds_cape = None
        try:
            ds_cape = xr.open_dataset(grib_file, engine='cfgrib', 
                                      filter_by_keys={'shortName': 'cape', 'typeOfLevel': 'surface'})
        except:
            pass # CAPE might be missing, handled later

        # --- 3. Load Updraft Helicity (0-3000m) ---
        ds_uh_max = None
        ds_uh_min = None
        
        # Try loading Max UH (mxuphl)
        try:
            # We filter loosely for mxuphl first
            ds_uh_raw = xr.open_dataset(grib_file, engine='cfgrib', 
                                       filter_by_keys={'shortName': 'mxuphl'})
            
            # Check if we have the 3000-0m layer
            # Usually labeled 'heightAboveGroundLayer' with topLevel=3000
            for var in ds_uh_raw.data_vars:
                da = ds_uh_raw[var]
                # Check for 0-3000m layer specifically
                if 'heightAboveGroundLayer' in da.coords:
                    # Some files might have 2-5km (5000-2000) or 0-3km (3000-0)
                    # We want the one where top is 3000
                    # Note: exact coord matching in cfgrib can be tricky, 
                    # so we assume if mxuphl exists in this file, it's likely the standard output.
                    ds_uh_max = da
                    print("       Found Max Updraft Helicity.")
                    break
        except Exception as e:
            print(f"       Max UH not found or error: {e}")

        # Try loading Min UH (mnuphl)
        try:
            ds_uh_min_raw = xr.open_dataset(grib_file, engine='cfgrib', 
                                           filter_by_keys={'shortName': 'mnuphl'})
            for var in ds_uh_min_raw.data_vars:
                ds_uh_min = ds_uh_min_raw[var]
                print("       Found Min Updraft Helicity.")
                break
        except:
            pass # Min UH is less common in some feeds

        # --- 4. Level Filtering ---
        file_levels = ds_wind.isobaricInhPa.values
        available_levels = sorted([l for l in REQUESTED_LEVELS if l in file_levels], reverse=True)
        if len(available_levels) < 3: return
        
        ds_wind = ds_wind.sel(isobaricInhPa=available_levels)
        u = ds_wind['u'].metpy.convert_units('kts')
        v = ds_wind['v'].metpy.convert_units('kts')
        level_values = available_levels

        # --- 5. Plotting Setup ---
        print(f"[f{fhr:02d}] Initializing Map...")
        fig = plt.figure(figsize=(18, 12))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal())
        ax.set_extent(REGION)
        
        ax.add_feature(cfeature.COASTLINE, linewidth=2.0, zorder=10)
        ax.add_feature(cfeature.BORDERS, linewidth=2.0, zorder=10)
        ax.add_feature(cfeature.STATES, linewidth=1.5, edgecolor='black', zorder=10)

        # --- 6. Plot CAPE (Background) ---
        if ds_cape is not None:
            cape_data = ds_cape['cape']
            cape_vals = cape_data.values
            cape_vals = np.where(cape_vals > 20000, 0, cape_vals)
            cape_vals = np.nan_to_num(cape_vals, nan=0.0)
            
            cape_plot = ax.contourf(cape_data.longitude, cape_data.latitude, cape_vals, 
                                    levels=CAPE_LEVELS, cmap=CAPE_CMAP, 
                                    extend='max', alpha=0.6, transform=ccrs.PlateCarree())
            
            ax.set_facecolor('white')
            plt.colorbar(cape_plot, ax=ax, orientation='horizontal', pad=0.02, 
                         aspect=50, shrink=0.8, label='SBCAPE (J/kg)')

        # --- 7. Plot UH Contours (Swaths) ---
        # Plot Max UH (Positive/Cyclonic)
        if ds_uh_max is not None:
            # Smooth slightly for cleaner contours
            uh_vals = ds_uh_max.values
            
            # Contours
            cs_max = ax.contour(ds_uh_max.longitude, ds_uh_max.latitude, uh_vals,
                               levels=UH_MAX_LEVELS, colors=UH_MAX_COLORS, 
                               linewidths=2.0, transform=ccrs.PlateCarree(), zorder=15)
            # Labels
            ax.clabel(cs_max, inline=True, fontsize=10, fmt='%d', colors='black', rightside_up=True)

        # Plot Min UH (Negative/Anti-Cyclonic)
        if ds_uh_min is not None:
            uh_min_vals = ds_uh_min.values
            cs_min = ax.contour(ds_uh_min.longitude, ds_uh_min.latitude, uh_min_vals,
                               levels=UH_MIN_LEVELS, colors=UH_MIN_COLORS, 
                               linewidths=1.5, linestyles='dashed', 
                               transform=ccrs.PlateCarree(), zorder=15)
            ax.clabel(cs_min, inline=True, fontsize=10, fmt='%d', colors='black')

        # --- 8. Legend ---
        legend_elements = [
            mlines.Line2D([], [], color='magenta', lw=3, label='0-1.5 km (>850mb)'),
            mlines.Line2D([], [], color='red', lw=3, label='1.5-3 km (850-700mb)'),
            mlines.Line2D([], [], color='green', lw=3, label='3-6 km (700-500mb)'),
            mlines.Line2D([], [], color='gold', lw=3, label='6-9 km (<500mb)'),
            # Legend for UH
            mlines.Line2D([], [], color='red', lw=2, label='0-3km UH (Max)'),
            mlines.Line2D([], [], color='blue', lw=2, linestyle='--', label='0-3km UH (Min)'),
            mlines.Line2D([], [], color='black', lw=0.5, alpha=0.5, label='Rings: 20 kts') 
        ]
        
        ax.legend(handles=legend_elements, loc='upper left', title="Hodograph & UH", 
                  framealpha=0.9, fontsize=11, title_fontsize=12).set_zorder(100)

        # --- 9. Plot Hodographs (Foreground) ---
        print(f"[f{fhr:02d}] Generating Colored Hodographs...")
        lons = u.longitude.values
        lats = u.latitude.values
        u_data = u.values
        v_data = v.values

        counter = 0
        for i in range(0, lons.shape[0], GRID_SPACING):
            for j in range(0, lons.shape[1], GRID_SPACING):
                
                if np.isnan(u_data[:, i, j]).any(): continue
                curr_lon = lons[i, j]
                curr_lat = lats[i, j]
                
                check_lon = curr_lon - 360 if curr_lon > 180 else curr_lon
                if not (REGION[0] < check_lon < REGION[1] and REGION[2] < curr_lat < REGION[3]):
                    continue

                try:
                    proj_pnt = ax.projection.transform_point(curr_lon, curr_lat, ccrs.PlateCarree())
                except: continue

                bounds = [proj_pnt[0] - BOX_SIZE/2, proj_pnt[1] - BOX_SIZE/2, BOX_SIZE, BOX_SIZE]
                sub_ax = ax.inset_axes(bounds, transform=ax.transData, zorder=20)
                
                h = Hodograph(sub_ax, component_range=80)
                h.add_grid(increment=20, color='black', alpha=0.3, linewidth=0.5)
                
                plot_colored_hodograph(h.ax, u_data[:, i, j], v_data[:, i, j], level_values)
                
                sub_ax.set_xticklabels([])
                sub_ax.set_yticklabels([])
                sub_ax.axis('off')
                counter += 1

        print(f"[f{fhr:02d}] Plotted {counter} hodographs.")

        # --- 10. Title ---
        valid_time = date_obj + timedelta(hours=fhr)
        valid_str = valid_time.strftime("%a %H:%MZ") 
        plt.title(f"HREF Mean CAPE & 0-3km UH | Run: {date_str} {run}Z | Valid: {valid_str} (f{fhr:02d})", 
                  fontsize=18, weight='bold')
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = f"{OUTPUT_DIR}/href_hodo_cape_{date_str}_{run}z_f{fhr:02d}.png"
        plt.savefig(out_path, bbox_inches='tight', dpi=90) 
        print(f"Saved: {out_path}")
        plt.close(fig)

    except Exception as e:
        print(f"Error processing f{fhr:02d}: {e}")
        traceback.print_exc()
    finally:
        if os.path.exists(grib_file):
            os.remove(grib_file)

if __name__ == "__main__":
    date_str, run, date_obj = get_latest_run_time()
    run_dt = datetime.datetime.strptime(f"{date_str} {run}", "%Y%m%d %H")
    print(f"Starting HREF Hodograph + CAPE + UH generation for {date_str} {run}Z")
    
    # Force update print to trigger git change
    print("Applying 0-3km UH Contours + Labels...")
    
    for fhr in range(1, 49):
        process_forecast_hour(run_dt, date_str, run, fhr)
