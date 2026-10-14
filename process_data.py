import os
import pandas as pd
import numpy as np

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

LOCATION_LATITUDES = {
    "Sokoto": 13.0059,
    "Kaduna": 10.5105,
    "Ibadan": 7.3775,
    "Port_Harcourt": 4.8156
}

def calculate_extraterrestrial_radiation(doy, lat_deg):
    """Calculates daily Extraterrestrial Radiation (Ra) in MJ/m^2/day (FAO-56 Eq. 21)."""
    lat_rad = np.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    sol_dec = 0.409 * np.sin((2 * np.pi * doy / 365) - 1.39)
    ws = np.arccos(-np.tan(lat_rad) * np.tan(sol_dec))
    
    # Solar constant Gsc = 0.0820 MJ/m^2/min
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(lat_rad) * np.sin(sol_dec) + 
        np.cos(lat_rad) * np.cos(sol_dec) * np.sin(ws)
    )
    return ra

def compute_eto_penman_monteith(df, lat_deg):
    """FAO-56 Penman-Monteith ET0 equation."""
    t_max = df["T2M_MAX"]
    t_min = df["T2M_MIN"]
    rh = df["RH2M"]
    rs = df["ALLSKY_SFC_SW_DWN"]
    u2 = df["WS2M"]
    
    t_mean = (t_max + t_min) / 2.0
    gamma = 0.066  # Psychrometric constant (kPa/°C)
    
    delta = (4098 * (0.6108 * np.exp((17.27 * t_mean) / (t_mean + 237.3)))) / ((t_mean + 237.3) ** 2)
    
    e_tmax = 0.6108 * np.exp((17.27 * t_max) / (t_max + 237.3))
    e_tmin = 0.6108 * np.exp((17.27 * t_min) / (t_min + 237.3))
    es = (e_tmax + e_tmin) / 2.0
    ea = (rh / 100.0) * es
    
    ra = calculate_extraterrestrial_radiation(df["DOY"], lat_deg)
    rso = (0.75 + 2e-5 * 100) * ra
    rns = (1 - 0.23) * rs
    
    sigma = 4.903e-9
    t_max_k4 = (t_max + 273.16) ** 4
    t_min_k4 = (t_min + 273.16) ** 4
    
    rs_rso_ratio = np.clip(rs / np.maximum(rso, 1e-5), 0.3, 1.0)
    rnl = sigma * ((t_max_k4 + t_min_k4) / 2.0) * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0))) * (1.35 * rs_rso_ratio - 0.35)
    
    rn = rns - rnl
    g = 0
    
    numerator = 0.408 * delta * (rn - g) + gamma * (900 / (t_mean + 273)) * u2 * (es - ea)
    denominator = delta + gamma * (1 + 0.34 * u2)
    
    eto_pm = numerator / denominator
    return np.maximum(eto_pm, 0)

def compute_eto_hargreaves(df, lat_deg):
    """Hargreaves-Samani ET0 formulation (1985)."""
    t_max = df["T2M_MAX"]
    t_min = df["T2M_MIN"]
    t_mean = (t_max + t_min) / 2.0
    tdiff = np.maximum(t_max - t_min, 0)
    
    ra = calculate_extraterrestrial_radiation(df["DOY"], lat_deg)
    eto_hs = 0.0023 * 0.408 * ra * (t_mean + 17.8) * np.sqrt(tdiff)
    return np.maximum(eto_hs, 0)

def process_raw_files():
    successful_count = 0
    for location, lat in LOCATION_LATITUDES.items():
        raw_file = os.path.join(RAW_DIR, f"{location}_raw_1990_2023.csv")
        if not os.path.exists(raw_file):
            print(f"File not found: {raw_file}")
            continue
            
        print(f"Processing data for {location}...")
        
        with open(raw_file, "r") as f:
            lines = f.readlines()

        header_idx = None
        for idx, line in enumerate(lines):
            clean_line = line.strip().upper()
            if "," in clean_line and "YEAR" in clean_line:
                header_idx = idx
                break

        if header_idx is None:
            print(f"Could not identify header index for {location}.")
            continue

        df = pd.read_csv(raw_file, skiprows=header_idx, on_bad_lines='skip')
        df.columns = [col.strip() for col in df.columns]
        
        # Standardize date column aliases across API variants
        rename_map = {"MM": "MO", "DD": "DY", "DAY": "DY", "MONTH": "MO", "JAN": "DOY"}
        df.rename(columns=rename_map, inplace=True)
        
        # Filter non-numeric rows
        df = df[pd.to_numeric(df['YEAR'], errors='coerce').notnull()].copy()
        df['YEAR'] = df['YEAR'].astype(int)
        
        # Robust Date Parsing (handles YEAR+DOY or YEAR+MO+DY)
        if "DOY" in df.columns:
            df['DOY'] = pd.to_numeric(df['DOY'], errors='coerce').astype(int)
            df['DATE'] = pd.to_datetime(df['YEAR'].astype(str), format='%Y') + pd.to_timedelta(df['DOY'] - 1, unit='D')
            df['MO'] = df['DATE'].dt.month
            df['DY'] = df['DATE'].dt.day
        elif "MO" in df.columns and "DY" in df.columns:
            df['MO'] = pd.to_numeric(df['MO'], errors='coerce').astype(int)
            df['DY'] = pd.to_numeric(df['DY'], errors='coerce').astype(int)
            df["DATE"] = pd.to_datetime(
                df["YEAR"].astype(str) + "-" + 
                df["MO"].astype(str).str.zfill(2) + "-" + 
                df["DY"].astype(str).str.zfill(2)
            )
            df["DOY"] = df["DATE"].dt.dayofyear
        else:
            print(f"Unable to parse date columns for {location}. Columns available: {list(df.columns)}")
            continue

        # Coerce meteorological variables
        for col in ["T2M_MAX", "T2M_MIN", "RH2M", "ALLSKY_SFC_SW_DWN", "WS2M"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Missing value handling (-999 flags)
        df.replace([-999, -999.0, -99], np.nan, inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        
        # Benchmark targets
        df["ET0_PM"] = compute_eto_penman_monteith(df, lat)
        df["ET0_HS"] = compute_eto_hargreaves(df, lat)
        df["LOCATION"] = location
        
        output_file = os.path.join(PROCESSED_DIR, f"{location}_processed.csv")
        df.to_csv(output_file, index=False)
        print(f"Successfully processed {len(df)} records -> {output_file}")
        successful_count += 1

    if successful_count == 0:
        print("\nNo locations were processed successfully.")

if __name__ == "__main__":
    process_raw_files()