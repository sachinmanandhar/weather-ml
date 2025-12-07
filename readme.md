# 🌦️ Weather Regime Clustering & Anomaly Detection

A comprehensive climate analysis pipeline for detecting extreme weather patterns in Nepal using unsupervised machine learning techniques.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Pipeline Steps](#pipeline-steps)
- [Weather Regime Clustering](#weather-regime-clustering)
- [Results](#results)
- [Output Files](#output-files)
- [Usage](#usage)
- [Future Enhancements](#future-enhancements)

---

## Overview

This project implements an unsupervised machine learning pipeline to:

1. **Cluster weather patterns** into distinct regimes (hot/dry, cold/dry, monsoon, etc.)
2. **Detect anomalies** in climate data to identify extreme weather events
3. **Analyze temporal patterns** in weather regime distributions

The analysis focuses on monthly climate data from Kathmandu, Nepal, spanning 2000-2023.

---

## Dataset

### Primary Data Sources

| File                        | Description                                      |
| --------------------------- | ------------------------------------------------ |
| `data/second/data.csv`      | Monthly climate observations (2000-2023)         |
| `data/second/parameter.csv` | Parameter metadata                               |
| `data/*.nc`                 | NetCDF ensemble forecast files (hourly, lat/lon) |

### Climate Parameters

| Parameter   | ID  | Description        |
| ----------- | --- | ------------------ |
| `temp`      | 8   | Temperature (°C)   |
| `rain`      | 7   | Rainfall (mm)      |
| `evap`      | 6   | Evaporation (mm)   |
| `soilMoist` | 9   | Soil Moisture (mm) |

### Data Structure

```
data.csv columns:
├── id              - Unique record identifier
├── year            - Year (2000-2023)
├── month           - Month (1-12)
├── param           - Parameter name
├── max             - Maximum value
├── min             - Minimum value
├── mean            - Mean value
├── district        - District name (Kathmandu)
├── state           - State/Province (Bagmati)
├── country         - Country code (NPL)
├── district_id_id  - District ID
└── parameter_id    - Parameter ID
```

---

## Installation

### Prerequisites

- Python 3.8+
- pipenv (recommended) or pip

### Setup

```bash
# Clone or navigate to project
cd /path/to/anamoly_detection

# Install dependencies with pipenv
pipenv install

# Or with pip
pip install pandas numpy scikit-learn matplotlib seaborn xarray netCDF4
```

### Required Libraries

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
xarray>=2023.1.0
netCDF4>=1.6.0
```

---

## Pipeline Steps

### Step 1: Data Loading & Preprocessing

```python
# Load CSV data
df = pd.read_csv("data/second/data.csv")

# Pivot to wide format (parameters as columns)
pivot_df = df.pivot_table(
    index=['year', 'month'],
    columns='param',
    values='mean'
)
```

### Step 2: Feature Engineering

Features used for clustering:

- **Temperature** (`temp`) - Primary thermal indicator
- **Rainfall** (`rain`) - Precipitation levels
- **Evaporation** (`evap`) - Moisture loss indicator
- **Soil Moisture** (`soilMoist`) - Ground water content

### Step 3: Standardization

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Step 4: K-Means Clustering

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
```

### Step 5: Regime Assignment

Clusters are labeled based on their meteorological characteristics:

| Cluster | Regime             | Criteria                             |
| ------- | ------------------ | ------------------------------------ |
| A       | Hot & Dry          | High temp, low rain, high evap       |
| B       | Cold & Dry         | Low temp, low rain                   |
| C       | Warm & Humid       | Moderate temp, high soil moisture    |
| D       | Extreme Rainfall   | Very high rain, high soil moisture   |
| E       | Pre-monsoon Storms | Moderate temp, transitional moisture |

---

## Weather Regime Clustering

### Methodology

The clustering uses **K-Means algorithm** with 5 clusters to identify distinct weather regimes:

```
┌─────────────────────────────────────────────────────────────┐
│                    CLUSTERING PIPELINE                       │
├─────────────────────────────────────────────────────────────┤
│  Raw Data → Pivot → Scale → K-Means → Label → Analyze       │
└─────────────────────────────────────────────────────────────┘
```

### Cluster Characteristics

| Regime                    | Temp (°C) | Rain (mm) | Evap (mm) | Soil Moist (mm) | Count |
| ------------------------- | --------- | --------- | --------- | --------------- | ----- |
| **A: Hot & Dry**          | 20.4      | 6.0       | 3.59      | 262.7           | 45    |
| **B: Cold & Dry**         | 9.5       | 0.4       | 1.27      | 222.9           | 98    |
| **C: Warm & Humid**       | 15.6      | 1.4       | 2.69      | 275.6           | 28    |
| **D: Extreme Rainfall**   | 20.1      | 12.4      | 3.53      | 323.0           | 56    |
| **E: Pre-monsoon Storms** | 17.5      | 1.7       | 1.95      | 200.7           | 55    |

### Seasonal Distribution

```
Month   │ Dominant Regime
────────┼──────────────────────────
Jan     │ B: Cold & Dry
Feb     │ B: Cold & Dry
Mar     │ E: Pre-monsoon Storms
Apr     │ E: Pre-monsoon Storms
May     │ A: Hot & Dry
Jun     │ A: Hot & Dry
Jul     │ D: Extreme Rainfall
Aug     │ D: Extreme Rainfall
Sep     │ D: Extreme Rainfall
Oct     │ C: Warm & Humid
Nov     │ B: Cold & Dry
Dec     │ B: Cold & Dry
```

---

## Results

### Visualization

The analysis produces a 4-panel visualization (`weather_regime_clustering.png`):

1. **Cluster Characteristics Heatmap** - Mean values per regime
2. **Monthly Distribution** - Stacked bar chart of regime percentages
3. **Temperature vs Rainfall Scatter** - Colored by weather regime
4. **Box Plot Distribution** - Temperature & rainfall by regime

### Key Findings

1. **Monsoon Season (Jul-Sep)**: Dominated by **Extreme Rainfall** regime

   - Average rainfall: 12.4 mm
   - Highest soil moisture: 323 mm

2. **Winter (Nov-Feb)**: Characterized by **Cold & Dry** conditions

   - Lowest temperatures: 9.5°C average
   - Minimal precipitation: 0.4 mm

3. **Pre-monsoon (Mar-Apr)**: Transitional **Pre-monsoon Storms** pattern

   - Building temperatures and moisture
   - Variable precipitation

4. **Early Summer (May-Jun)**: **Hot & Dry** regime before monsoon onset
   - Peak temperatures: 20.4°C
   - High evaporation rates

---

## Output Files

| File                            | Description                                    |
| ------------------------------- | ---------------------------------------------- |
| `weather_regime_clustering.png` | 4-panel visualization of clustering results    |
| `weather_regime_results.csv`    | Full dataset with cluster & regime assignments |
| `anomaly_detection_results.png` | Anomaly detection visualization (if run)       |

### Output CSV Structure

```csv
year,month,evap,rain,soilMoist,temp,cluster,weather_regime
2000,1,0.94,0.25,221.42,3.49,3,B: Cold & Dry
2000,2,1.04,0.31,216.51,3.38,3,B: Cold & Dry
...
```

---

## Usage

### Run Weather Regime Clustering

```bash
# Activate environment
pipenv shell

# Run clustering analysis
python second.py
```

### Run NetCDF Analysis (Optional)

```bash
# For NetCDF-based analysis
python index.py
```

### Run Tests

```bash
python test_index.py
```

---

## Future Enhancements

### 1. Anomaly Detection within Clusters

```python
from sklearn.ensemble import IsolationForest

# Detect anomalies within each weather regime
iso_forest = IsolationForest(contamination=0.1, random_state=42)
anomalies = iso_forest.fit_predict(X_scaled)
```

### 2. PCA Dimensionality Reduction

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

### 3. One-Class SVM for Extreme Event Detection

```python
from sklearn.svm import OneClassSVM

oc_svm = OneClassSVM(kernel='rbf', nu=0.1)
predictions = oc_svm.fit_predict(X_scaled)
```

### 4. Temporal Smoothing

```python
# Rolling mean for noise reduction
df['temp_smooth'] = df['temp'].rolling(window=3).mean()
```

### 5. Multi-Model Comparison

Compare anomaly scores from:

- Isolation Forest
- One-Class SVM
- Local Outlier Factor (LOF)
- DBSCAN

### 6. Spatial Analysis

Map anomalies on Nepal grid using lat/lon coordinates from NetCDF files.

---

## Project Structure

```
anamoly_detection/
├── data/
│   ├── second/
│   │   ├── data.csv           # Monthly climate data
│   │   └── parameter.csv      # Parameter metadata
│   └── *.nc                   # NetCDF ensemble files
├── index.py                   # NetCDF analysis script
├── second.py                  # Weather regime clustering
├── test_index.py              # Unit tests
├── readme.md                  # This file
├── Pipfile                    # Python dependencies
├── Pipfile.lock               # Locked dependencies
├── weather_regime_clustering.png    # Clustering visualization
├── weather_regime_results.csv       # Clustering results
└── anomaly_detection_results.png    # Anomaly visualization
```

---

## References

- **K-Means Clustering**: MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations.
- **Isolation Forest**: Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest.
- **Climate Data**: Nepal Department of Hydrology and Meteorology

---

## License

This project is for research and educational purposes.

---

## Contact

For questions or contributions, please open an issue in the repository.
