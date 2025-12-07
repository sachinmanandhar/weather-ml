import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Load multiple NetCDF files and combine
def load_multiple_files(data_dir="data", pattern="hkhEnsemble_*_hourly_latlon.nc"):
    """Load all matching NetCDF files and combine them"""
    data_path = Path(data_dir)
    files = sorted(data_path.glob(pattern))
    
    datasets = []
    for file in files:
        ds = xr.open_dataset(file)
        ds = ds[['ensmean-tmp2m']]
        datasets.append(ds)
    
    # Combine all datasets along time dimension
    combined = xr.concat(datasets, dim='time')
    combined = combined.sortby('time')  # Ensure chronological order
    return combined

# 2. Extract time series (you can choose specific location or aggregate)
def extract_timeseries(ds, lat=None, lon=None, method='mean'):
    """
    Extract time series from dataset
    - If lat/lon specified: get time series at that location
    - If method='mean': get spatial average time series
    - If method='max' or 'min': get spatial max/min time series
    """
    if lat is not None and lon is not None:
        # Get time series at specific location
        ts = ds['ensmean-tmp2m'].sel(latitude=lat, longitude=lon, method='nearest')
    else:
        # Get spatial aggregate
        if method == 'mean':
            ts = ds['ensmean-tmp2m'].mean(dim=['latitude', 'longitude'])
        elif method == 'max':
            ts = ds['ensmean-tmp2m'].max(dim=['latitude', 'longitude'])
        elif method == 'min':
            ts = ds['ensmean-tmp2m'].min(dim=['latitude', 'longitude'])
    
    return ts

# 3. Anomaly detection methods
def detect_anomalies_statistical(ts, threshold=3):
    """Detect anomalies using Z-score method"""
    values = ts.values
    mean = np.nanmean(values)
    std = np.nanstd(values)
    
    z_scores = np.abs((values - mean) / std)
    anomalies = z_scores > threshold
    
    return anomalies, z_scores

def detect_anomalies_isolation_forest(ts, contamination=0.1):
    """Detect anomalies using Isolation Forest"""
    values = ts.values.reshape(-1, 1)
    
    # Handle NaN values
    mask = ~np.isnan(values.flatten())
    clean_values = values[mask].reshape(-1, 1)
    
    # Fit Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    predictions = np.full(len(values), 1)  # 1 = normal, -1 = anomaly
    predictions[mask] = iso_forest.fit_predict(clean_values)
    
    anomalies = predictions == -1
    return anomalies, iso_forest.score_samples(values)

def detect_anomalies_rolling_window(ts, window=24, threshold=2):
    """Detect anomalies using rolling window statistics"""
    values = ts.values
    df = pd.Series(values, index=ts.time.values)
    
    # Calculate rolling mean and std
    rolling_mean = df.rolling(window=window, center=True).mean()
    rolling_std = df.rolling(window=window, center=True).std()
    
    # Detect points outside rolling window
    z_scores = np.abs((df - rolling_mean) / rolling_std)
    anomalies = z_scores > threshold
    
    return anomalies.values, z_scores.values

# 4. Main workflow
if __name__ == "__main__":
    # Load all files
    print("Loading NetCDF files...")
    ds = load_multiple_files()
    print(f"Combined dataset: {ds}")
    
    # Extract time series (choose one method)
    print("\nExtracting time series...")
    # Option 1: Spatial average
    ts = extract_timeseries(ds, method='mean')
    
    # Option 2: Specific location (e.g., Kathmandu area: ~27.7°N, 85.3°E)
    # ts = extract_timeseries(ds, lat=27.7, lon=85.3)
    
    print(f"Time series shape: {ts.shape}")
    print(f"Time range: {ts.time.min().values} to {ts.time.max().values}")
    
    # Detect anomalies using multiple methods
    print("\nDetecting anomalies...")
    
    # Method 1: Statistical (Z-score)
    anomalies_zscore, z_scores = detect_anomalies_statistical(ts, threshold=3)
    print(f"Z-score method: {anomalies_zscore.sum()} anomalies detected")
    
    # Method 2: Isolation Forest
    anomalies_iso, iso_scores = detect_anomalies_isolation_forest(ts, contamination=0.1)
    print(f"Isolation Forest: {anomalies_iso.sum()} anomalies detected")
    
    # Method 3: Rolling window (good for time series with patterns)
    anomalies_rolling, rolling_scores = detect_anomalies_rolling_window(ts, window=24, threshold=2)
    print(f"Rolling window: {anomalies_rolling.sum()} anomalies detected")
    
    # Visualize
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Time series with anomalies
    plt.subplot(3, 1, 1)
    plt.plot(ts.time.values, ts.values, label='Temperature', alpha=0.7)
    plt.scatter(ts.time.values[anomalies_zscore], ts.values[anomalies_zscore], 
                color='red', label='Z-score Anomalies', zorder=5)
    plt.scatter(ts.time.values[anomalies_iso], ts.values[anomalies_iso], 
                color='orange', marker='x', label='Isolation Forest Anomalies', zorder=5)
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.title('Time Series with Detected Anomalies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Z-scores
    plt.subplot(3, 1, 2)
    plt.plot(ts.time.values, z_scores, label='Z-scores')
    plt.axhline(y=3, color='r', linestyle='--', label='Threshold (3σ)')
    plt.xlabel('Time')
    plt.ylabel('Z-score')
    plt.title('Z-score Anomaly Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Anomaly scores comparison
    plt.subplot(3, 1, 3)
    plt.plot(ts.time.values, -iso_scores, label='Isolation Forest Score', alpha=0.7)
    plt.plot(ts.time.values, rolling_scores, label='Rolling Window Z-score', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Anomaly Score')
    plt.title('Anomaly Scores Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('anomaly_detection_results.png', dpi=150)
    print("\nResults saved to 'anomaly_detection_results.png'")
    
    # Print anomaly timestamps
    print("\nAnomaly Timestamps (Z-score method):")
    anomaly_times = ts.time.values[anomalies_zscore]
    for t in anomaly_times:
        print(f"  {t}")