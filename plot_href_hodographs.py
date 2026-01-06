import xarray as xr
import matplotlib.pyplot as plt
import requests
import os
import sys
import datetime
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
REQUESTED_LEVELS = [1000, 925, 850, 700, 500, 250]

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
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    try:
        with requests.get(url, stream=True, timeout=60, headers=headers) as r:
            if r.status_code == 404:
                print(f"File not found. Forecast f{fhr:02d} may not be ready yet.")
                return None
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return filename
    except Exception as e:
        print(f"Download failed: {e}")
        return None

def diagnose_grib_file(grib_file):
    print("\n" + "="*60)
    print(f"DIAGNOSTIC MODE: Scanning {grib_file}...")
    print("="*60)
    
    try:
        # open_datasets (PLURAL) forces it to read all distinct message types
        datasets = xr.open_datasets(grib_file, engine='cfgrib')
        
        found_uh = False
        
        for i, ds in enumerate(datasets):
            print(f"\n--- Dataset Part {i+1} ---")
            for var_name in ds.data_vars:
                da = ds[var_name]
                attrs = da.attrs
                
                short_name = attrs.get('GRIB_shortName', 'unknown')
                long_name = attrs.get('long_name', 'unknown')
                level_type = attrs.get('GRIB_typeOfLevel', 'unknown')
                
                # Try to extract level info if it exists
                # Different GRIB keys might hold the height info
                level_desc = "N/A"
                if 'heightAboveGroundLayer' in da.coords:
                    level_desc = str(da['heightAboveGroundLayer'].values)
                elif 'isobaricInhPa' in da.coords:
                    level_desc = f"{da['isobaricInhPa'].values} mb"
                elif 'surface' in da.coords:
                    level_desc = "Surface"
                elif 'heightAboveGround' in da.coords:
                    level_desc = f"{da['heightAboveGround'].values} m"
                
                # Check for Helicity keywords
                is_helicity = 'helicity' in long_name.lower() or 'uphl' in short_name.lower()
                prefix = ">>> FOUND POSSIBLE MATCH: " if is_helicity else "    "
                
                print(f"{prefix}Var: {var_name:<10} | Short: {short_name:<10} | Level: {level_type:<25} | Range: {level_desc}")

        print("\n" + "="*60)
        print("DIAGNOSTIC COMPLETE. Copy the lines above starting with '>>>' and paste them into the chat.")
        print("="*60)

    except Exception as e:
        print(f"Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

def process_forecast_hour(date_obj, date_str, run, fhr):
    grib_file = download_href_mean(date_str, run, fhr)
    if not grib_file:
        return

    # RUN DIAGNOSTIC AND QUIT
    diagnose_grib_file(grib_file)
    
    # Clean up and exit so the log isn't huge
    if os.path.exists(grib_file):
        os.remove(grib_file)
    sys.exit(0) 

if __name__ == "__main__":
    date_str, run, date_obj = get_latest_run_time()
    run_dt = datetime.datetime.strptime(f"{date_str} {run}", "%Y%m%d %H")
    print(f"Starting DIAGNOSTIC RUN for {date_str} {run}Z")
    
    # We only need to check ONE file to find the variable names
    # Usually f01 or f02 has everything.
    process_forecast_hour(run_dt, date_str, run, 1)
