import xarray as xr
from pathlib import Path


def load_multiple_files(data_dir="data",pattern="hkhEnsemble_*_hourly_latlon.nc"):
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

# 4. Main workflow
if __name__ == "__main__":
    # Load all files
    print("Loading NetCDF files...")
    ds = load_multiple_files()
    print(f"Combined dataset: {ds}")
