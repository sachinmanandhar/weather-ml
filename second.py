"""
Weather Regime Clustering Analysis with Trend & Anomaly Detection
==================================================================
Enhanced Pipeline:
- Cluster monthly climate data into 5 weather regimes
- Detect trends using linear regression and Mann-Kendall test
- Identify anomalies using Isolation Forest
- Generate data-driven decision support recommendations

Pipeline: Raw Data → Standardize → PCA → K-Means → Label → Trend → Anomaly → Decisions
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful visualizations
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#C73E1D',
    'warning': '#E94F37',
    'anomaly': '#E63946',
    'normal': '#457B9D',
    'trend_up': '#E07A5F',
    'trend_down': '#3D405B',
    'neutral': '#81B29A'
}

# ============================================================
# DATA LOADING
# ============================================================
print("=" * 70)
print("  WEATHER REGIME ANALYSIS WITH TREND & ANOMALY DETECTION")
print("  Decision Support System for Climate-Smart Planning")
print("=" * 70)

data_path = "data/second/data.csv"
param_path = "data/second/parameter.csv"

df = pd.read_csv(data_path)
params = pd.read_csv(param_path)

print(f"\n📊 Data Overview:")
print(f"   Shape: {df.shape}")
print(f"   Parameters: {params['param'].tolist()}")
print(f"   Year range: {df['year'].min()} - {df['year'].max()}")
print(f"   Unique months: {sorted(df['month'].unique())}")

# ============================================================
# STEP 1: DATA PIVOTING & PREPARATION
# ============================================================
pivot_df = df.pivot_table(
    index=['year', 'month'],
    columns='param',
    values='mean',
    aggfunc='first'
).reset_index()

# Create datetime column for trend analysis
pivot_df['date'] = pd.to_datetime(pivot_df['year'].astype(str) + '-' + 
                                   pivot_df['month'].astype(str) + '-01')
pivot_df['time_index'] = (pivot_df['year'] - pivot_df['year'].min()) * 12 + pivot_df['month']

print(f"\n📈 Pivoted data shape: {pivot_df.shape}")

# ============================================================
# STEP 2: CLUSTERING (Existing Pipeline)
# ============================================================
print("\n" + "=" * 70)
print("  PHASE 1: WEATHER REGIME CLUSTERING")
print("=" * 70)

feature_cols = ['evap', 'rain', 'soilMoist', 'temp']
X = pivot_df[feature_cols].copy()
X = X.dropna()
pivot_df_clean = pivot_df.loc[X.index].copy()

# Standardize and apply PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"   ✓ Standardized {len(X)} data points")
print(f"   ✓ PCA variance explained: {sum(pca.explained_variance_ratio_)*100:.1f}%")

# K-Means Clustering
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

pivot_df_clean['cluster'] = clusters
pivot_df_clean['PC1'] = X_pca[:, 0]
pivot_df_clean['PC2'] = X_pca[:, 1]

# Assign weather regime labels
cluster_stats = pivot_df_clean.groupby('cluster')[feature_cols].mean()

def assign_weather_regime(cluster_stats):
    labels = {}
    norm_stats = (cluster_stats - cluster_stats.min()) / (cluster_stats.max() - cluster_stats.min())
    
    scores = pd.DataFrame(index=cluster_stats.index)
    scores['hot_dry'] = norm_stats['temp'] * 0.4 + (1 - norm_stats['rain']) * 0.4 + norm_stats['evap'] * 0.2
    scores['cold_dry'] = (1 - norm_stats['temp']) * 0.5 + (1 - norm_stats['rain']) * 0.5
    scores['warm_humid'] = norm_stats['temp'] * 0.3 + norm_stats['soilMoist'] * 0.4 + norm_stats['rain'] * 0.3
    scores['extreme_rain'] = norm_stats['rain'] * 0.7 + norm_stats['soilMoist'] * 0.3
    scores['pre_monsoon'] = norm_stats['temp'] * 0.3 + norm_stats['rain'] * 0.3 + (1 - norm_stats['soilMoist']) * 0.2 + norm_stats['evap'] * 0.2
    
    regime_names = {
        'hot_dry': 'A: Hot & Dry',
        'cold_dry': 'B: Cold & Dry',
        'warm_humid': 'C: Warm & Humid',
        'extreme_rain': 'D: Extreme Rainfall',
        'pre_monsoon': 'E: Pre-monsoon Storms'
    }
    
    assigned = set()
    for _ in range(n_clusters):
        best_score = -1
        best_cluster = None
        best_regime = None
        
        for cluster in cluster_stats.index:
            if cluster in labels:
                continue
            for regime in regime_names.keys():
                if regime in assigned:
                    continue
                if scores.loc[cluster, regime] > best_score:
                    best_score = scores.loc[cluster, regime]
                    best_cluster = cluster
                    best_regime = regime
        
        if best_cluster is not None and best_regime is not None:
            labels[best_cluster] = regime_names[best_regime]
            assigned.add(best_regime)
    
    return labels

regime_labels = assign_weather_regime(cluster_stats)
pivot_df_clean['weather_regime'] = pivot_df_clean['cluster'].map(regime_labels)

print(f"   ✓ Identified {n_clusters} weather regimes")
for cluster, label in sorted(regime_labels.items()):
    count = (pivot_df_clean['cluster'] == cluster).sum()
    print(f"      Cluster {cluster} → {label} ({count} months)")

# ============================================================
# STEP 3: TREND ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("  PHASE 2: TREND ANALYSIS")
print("=" * 70)

def calculate_trend(data, time_index):
    """Calculate linear trend with statistical significance"""
    slope, intercept, r_value, p_value, std_err = stats.linregress(time_index, data)
    
    # Mann-Kendall trend test (simplified)
    n = len(data)
    s = 0
    for i in range(n-1):
        for j in range(i+1, n):
            s += np.sign(data.iloc[j] - data.iloc[i])
    
    var_s = (n * (n - 1) * (2 * n + 5)) / 18
    if s > 0:
        z_mk = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z_mk = (s + 1) / np.sqrt(var_s)
    else:
        z_mk = 0
    
    mk_p_value = 2 * (1 - stats.norm.cdf(abs(z_mk)))
    
    return {
        'slope': slope,
        'r_squared': r_value**2,
        'p_value': p_value,
        'trend_per_decade': slope * 120,  # 12 months * 10 years
        'mk_z': z_mk,
        'mk_p_value': mk_p_value,
        'significant': p_value < 0.05
    }

# Calculate trends for each parameter
trend_results = {}
yearly_avg = pivot_df_clean.groupby('year')[feature_cols].mean().reset_index()

for param in feature_cols:
    trend = calculate_trend(yearly_avg[param], yearly_avg['year'])
    trend_results[param] = trend
    
    direction = "↑ INCREASING" if trend['slope'] > 0 else "↓ DECREASING"
    sig = "***" if trend['significant'] else ""
    
    print(f"\n   📊 {param.upper()}")
    print(f"      Trend: {direction} {sig}")
    print(f"      Change per decade: {trend['trend_per_decade']:.3f}")
    print(f"      R²: {trend['r_squared']:.3f}, p-value: {trend['p_value']:.4f}")

# ============================================================
# STEP 4: ANOMALY DETECTION WITH ISOLATION FOREST
# ============================================================
print("\n" + "=" * 70)
print("  PHASE 3: ANOMALY DETECTION (Isolation Forest)")
print("=" * 70)

# Prepare features for anomaly detection
X_anomaly = pivot_df_clean[feature_cols].copy()
X_anomaly_scaled = scaler.transform(X_anomaly)

# Apply Isolation Forest
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.1,  # Expect ~10% anomalies
    random_state=42,
    max_samples='auto'
)

anomaly_labels = iso_forest.fit_predict(X_anomaly_scaled)
anomaly_scores = iso_forest.decision_function(X_anomaly_scaled)

pivot_df_clean['anomaly_label'] = anomaly_labels
pivot_df_clean['anomaly_score'] = anomaly_scores
pivot_df_clean['is_anomaly'] = anomaly_labels == -1

# Classify anomaly types
def classify_anomaly(row, means, stds):
    """Classify what type of anomaly this is"""
    deviations = {}
    for param in feature_cols:
        z_score = (row[param] - means[param]) / stds[param]
        if abs(z_score) > 2:
            direction = "high" if z_score > 0 else "low"
            deviations[param] = (z_score, direction)
    return deviations

means = pivot_df_clean[feature_cols].mean()
stds = pivot_df_clean[feature_cols].std()

anomalies = pivot_df_clean[pivot_df_clean['is_anomaly']].copy()
anomaly_types = []

for idx, row in anomalies.iterrows():
    deviations = classify_anomaly(row, means, stds)
    if deviations:
        most_extreme = max(deviations.items(), key=lambda x: abs(x[1][0]))
        anomaly_types.append({
            'year': int(row['year']),
            'month': int(row['month']),
            'primary_cause': f"{most_extreme[0]} ({most_extreme[1][1]})",
            'z_score': most_extreme[1][0],
            'all_deviations': deviations
        })

print(f"\n   🔍 Detection Results:")
print(f"      Total data points: {len(pivot_df_clean)}")
print(f"      Anomalies detected: {len(anomalies)} ({len(anomalies)/len(pivot_df_clean)*100:.1f}%)")

print(f"\n   📋 Top Anomalies by Severity:")
sorted_anomalies = sorted(anomaly_types, key=lambda x: abs(x['z_score']), reverse=True)[:10]
for i, anom in enumerate(sorted_anomalies, 1):
    print(f"      {i}. {anom['year']}-{anom['month']:02d}: {anom['primary_cause']} (z={anom['z_score']:.2f})")

# Anomaly patterns by month
monthly_anomaly_rate = pivot_df_clean.groupby('month')['is_anomaly'].mean() * 100
print(f"\n   📅 Monthly Anomaly Rates:")
month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
               7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
for month in range(1, 13):
    if month in monthly_anomaly_rate.index:
        rate = monthly_anomaly_rate[month]
        bar = "█" * int(rate / 5) + "░" * (10 - int(rate / 5))
        print(f"      {month_names[month]}: {bar} {rate:.1f}%")

# ============================================================
# STEP 5: DECISION SUPPORT SYSTEM
# ============================================================
print("\n" + "=" * 70)
print("  PHASE 4: DECISION SUPPORT RECOMMENDATIONS")
print("=" * 70)

def generate_decisions():
    """Generate actionable recommendations based on analysis"""
    decisions = {
        'agriculture': [],
        'water_management': [],
        'urban_planning': [],
        'disaster_preparedness': [],
        'health_sector': []
    }
    
    # Based on trends
    if trend_results['temp']['significant'] and trend_results['temp']['slope'] > 0:
        decisions['agriculture'].append({
            'priority': 'HIGH',
            'finding': f"Temperature increasing by {trend_results['temp']['trend_per_decade']:.2f}°C per decade",
            'recommendation': "Shift to heat-tolerant crop varieties; adjust planting calendars",
            'timeline': 'Immediate planning, implement within 1-2 growing seasons'
        })
        decisions['health_sector'].append({
            'priority': 'HIGH',
            'finding': "Warming trend detected",
            'recommendation': "Expand heat-health warning systems; increase cooling centers",
            'timeline': 'Before next summer season'
        })
    
    if trend_results['rain']['significant']:
        if trend_results['rain']['slope'] < 0:
            decisions['water_management'].append({
                'priority': 'CRITICAL',
                'finding': f"Rainfall declining by {abs(trend_results['rain']['trend_per_decade']):.2f}mm per decade",
                'recommendation': "Invest in water storage infrastructure; implement water recycling",
                'timeline': 'Begin infrastructure planning immediately'
            })
            decisions['agriculture'].append({
                'priority': 'HIGH',
                'finding': "Declining precipitation trend",
                'recommendation': "Adopt drip irrigation; select drought-resistant varieties",
                'timeline': 'Implement in next growing season'
            })
        else:
            decisions['disaster_preparedness'].append({
                'priority': 'HIGH',
                'finding': f"Rainfall increasing by {trend_results['rain']['trend_per_decade']:.2f}mm per decade",
                'recommendation': "Upgrade drainage systems; strengthen flood early warning",
                'timeline': 'Complete upgrades before monsoon'
            })
    
    if trend_results['soilMoist']['significant'] and trend_results['soilMoist']['slope'] < 0:
        decisions['agriculture'].append({
            'priority': 'HIGH',
            'finding': "Declining soil moisture trend",
            'recommendation': "Implement mulching and cover cropping; improve soil organic matter",
            'timeline': 'Start soil improvement programs within 6 months'
        })
    
    # Based on anomaly patterns
    high_anomaly_months = [m for m in range(1, 13) 
                          if m in monthly_anomaly_rate.index and monthly_anomaly_rate[m] > 15]
    
    if high_anomaly_months:
        decisions['disaster_preparedness'].append({
            'priority': 'MEDIUM',
            'finding': f"High anomaly rates in months: {[month_names[m] for m in high_anomaly_months]}",
            'recommendation': "Enhanced monitoring during these months; pre-position emergency resources",
            'timeline': 'Update protocols before identified months'
        })
    
    # Based on regime analysis
    extreme_rain_pct = (pivot_df_clean['weather_regime'] == 'D: Extreme Rainfall').mean() * 100
    if extreme_rain_pct > 10:
        decisions['urban_planning'].append({
            'priority': 'HIGH',
            'finding': f"Extreme rainfall regime occurs {extreme_rain_pct:.1f}% of the time",
            'recommendation': "Review and expand urban drainage capacity; restrict development in flood zones",
            'timeline': 'Include in next urban development plan'
        })
    
    hot_dry_pct = (pivot_df_clean['weather_regime'] == 'A: Hot & Dry').mean() * 100
    if hot_dry_pct > 15:
        decisions['water_management'].append({
            'priority': 'MEDIUM',
            'finding': f"Hot & dry conditions occur {hot_dry_pct:.1f}% of the time",
            'recommendation': "Develop water rationing protocols; promote rainwater harvesting",
            'timeline': 'Policy ready before dry season'
        })
    
    return decisions

decisions = generate_decisions()

print("\n   📋 SECTOR-WISE RECOMMENDATIONS:\n")

sector_icons = {
    'agriculture': '🌾',
    'water_management': '💧',
    'urban_planning': '🏙️',
    'disaster_preparedness': '🚨',
    'health_sector': '🏥'
}

for sector, recommendations in decisions.items():
    if recommendations:
        print(f"\n   {sector_icons[sector]} {sector.upper().replace('_', ' ')}:")
        print("   " + "-" * 60)
        for rec in recommendations:
            print(f"\n      [{rec['priority']}] {rec['finding']}")
            print(f"      → Action: {rec['recommendation']}")
            print(f"      → Timeline: {rec['timeline']}")

# ============================================================
# STEP 6: CREATE COMPREHENSIVE VISUALIZATIONS
# ============================================================
print("\n\n" + "=" * 70)
print("  PHASE 5: GENERATING VISUALIZATIONS")
print("=" * 70)

fig = plt.figure(figsize=(20, 24))
fig.suptitle('Weather Analysis: Trends, Anomalies & Decision Support\nKathmandu Climate Intelligence Dashboard', 
             fontsize=16, fontweight='bold', y=0.995)

# --- Row 1: Trend Analysis ---
# Plot 1: Temperature Trend
ax1 = fig.add_subplot(4, 3, 1)
yearly_avg = pivot_df_clean.groupby('year')[feature_cols].mean().reset_index()
ax1.scatter(yearly_avg['year'], yearly_avg['temp'], color=COLORS['primary'], alpha=0.7, s=50)
z = np.polyfit(yearly_avg['year'], yearly_avg['temp'], 1)
p = np.poly1d(z)
ax1.plot(yearly_avg['year'], p(yearly_avg['year']), color=COLORS['trend_up'], linewidth=2, 
         label=f'Trend: {z[0]*10:.2f}°C/decade')
ax1.fill_between(yearly_avg['year'], yearly_avg['temp'], p(yearly_avg['year']), alpha=0.2, color=COLORS['primary'])
ax1.set_xlabel('Year')
ax1.set_ylabel('Temperature (°C)')
ax1.set_title('🌡️ Temperature Trend', fontweight='bold')
ax1.legend(loc='upper left')
sig_text = "✓ Significant" if trend_results['temp']['significant'] else "○ Not Significant"
ax1.text(0.98, 0.02, sig_text, transform=ax1.transAxes, ha='right', fontsize=9, 
         color='green' if trend_results['temp']['significant'] else 'gray')

# Plot 2: Rainfall Trend
ax2 = fig.add_subplot(4, 3, 2)
ax2.scatter(yearly_avg['year'], yearly_avg['rain'], color=COLORS['secondary'], alpha=0.7, s=50)
z = np.polyfit(yearly_avg['year'], yearly_avg['rain'], 1)
p = np.poly1d(z)
ax2.plot(yearly_avg['year'], p(yearly_avg['year']), color=COLORS['trend_down'], linewidth=2,
         label=f'Trend: {z[0]*10:.2f}mm/decade')
ax2.fill_between(yearly_avg['year'], yearly_avg['rain'], p(yearly_avg['year']), alpha=0.2, color=COLORS['secondary'])
ax2.set_xlabel('Year')
ax2.set_ylabel('Rainfall (mm)')
ax2.set_title('🌧️ Rainfall Trend', fontweight='bold')
ax2.legend(loc='upper left')
sig_text = "✓ Significant" if trend_results['rain']['significant'] else "○ Not Significant"
ax2.text(0.98, 0.02, sig_text, transform=ax2.transAxes, ha='right', fontsize=9,
         color='green' if trend_results['rain']['significant'] else 'gray')

# Plot 3: All Parameters Trend Summary
ax3 = fig.add_subplot(4, 3, 3)
params_display = ['Temperature', 'Rainfall', 'Soil Moisture', 'Evaporation']
trends = [trend_results[p]['trend_per_decade'] for p in feature_cols]
colors_bar = [COLORS['trend_up'] if t > 0 else COLORS['trend_down'] for t in trends]
significance = ['*' if trend_results[p]['significant'] else '' for p in feature_cols]
bars = ax3.barh(params_display, [abs(t) for t in trends], color=colors_bar, alpha=0.8)
for i, (bar, trend, sig) in enumerate(zip(bars, trends, significance)):
    direction = '→ +' if trend > 0 else '→ -'
    ax3.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
             f'{direction}{abs(trend):.2f}{sig}', va='center', fontsize=9)
ax3.set_xlabel('Change per Decade (absolute)')
ax3.set_title('📊 Trend Summary (* = significant)', fontweight='bold')
ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

# --- Row 2: Anomaly Detection ---
# Plot 4: Anomaly Score Time Series
ax4 = fig.add_subplot(4, 3, 4)
normal_data = pivot_df_clean[~pivot_df_clean['is_anomaly']]
anomaly_data = pivot_df_clean[pivot_df_clean['is_anomaly']]
ax4.scatter(normal_data['date'], normal_data['anomaly_score'], c=COLORS['normal'], 
            alpha=0.5, s=20, label='Normal')
ax4.scatter(anomaly_data['date'], anomaly_data['anomaly_score'], c=COLORS['anomaly'], 
            alpha=0.8, s=50, marker='X', label='Anomaly')
ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax4.set_xlabel('Date')
ax4.set_ylabel('Anomaly Score')
ax4.set_title('🔍 Anomaly Detection Timeline', fontweight='bold')
ax4.legend()

# Plot 5: Anomaly Distribution by Month
ax5 = fig.add_subplot(4, 3, 5)
monthly_counts = pivot_df_clean.groupby('month').agg({
    'is_anomaly': ['sum', 'count']
}).reset_index()
monthly_counts.columns = ['month', 'anomalies', 'total']
monthly_counts['rate'] = monthly_counts['anomalies'] / monthly_counts['total'] * 100

colors_month = [COLORS['anomaly'] if r > 10 else COLORS['normal'] for r in monthly_counts['rate']]
bars = ax5.bar([month_names[m] for m in monthly_counts['month']], 
               monthly_counts['rate'], color=colors_month, alpha=0.8)
ax5.axhline(y=10, color=COLORS['warning'], linestyle='--', label='10% threshold')
ax5.set_xlabel('Month')
ax5.set_ylabel('Anomaly Rate (%)')
ax5.set_title('📅 Monthly Anomaly Patterns', fontweight='bold')
ax5.legend()
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

# Plot 6: Anomalies in Feature Space
ax6 = fig.add_subplot(4, 3, 6)
scatter_normal = ax6.scatter(normal_data['temp'], normal_data['rain'], 
                              c=COLORS['normal'], alpha=0.4, s=30, label='Normal')
scatter_anomaly = ax6.scatter(anomaly_data['temp'], anomaly_data['rain'], 
                               c=COLORS['anomaly'], alpha=0.8, s=80, marker='X', 
                               edgecolors='black', linewidth=0.5, label='Anomaly')
ax6.set_xlabel('Temperature (°C)')
ax6.set_ylabel('Rainfall (mm)')
ax6.set_title('🎯 Anomalies in Temp-Rain Space', fontweight='bold')
ax6.legend()

# --- Row 3: Clustering & Decision Support ---
# Plot 7: Clusters in PCA Space with Anomalies
ax7 = fig.add_subplot(4, 3, 7)
regime_colors = {'A: Hot & Dry': '#FF6B6B', 'B: Cold & Dry': '#4ECDC4', 
                 'C: Warm & Humid': '#45B7D1', 'D: Extreme Rainfall': '#96CEB4',
                 'E: Pre-monsoon Storms': '#FFEAA7'}

for regime in sorted(pivot_df_clean['weather_regime'].unique()):
    mask = (pivot_df_clean['weather_regime'] == regime) & (~pivot_df_clean['is_anomaly'])
    ax7.scatter(pivot_df_clean.loc[mask, 'PC1'], pivot_df_clean.loc[mask, 'PC2'],
                label=regime, alpha=0.6, c=regime_colors.get(regime, 'gray'), s=40)

# Overlay anomalies
ax7.scatter(anomaly_data['PC1'], anomaly_data['PC2'], 
            c='red', marker='X', s=100, edgecolors='black', 
            linewidth=1, label='ANOMALY', zorder=5)

ax7.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax7.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax7.set_title('🌐 Weather Regimes + Anomalies', fontweight='bold')
ax7.legend(fontsize=7, loc='upper right')

# Plot 8: Regime Distribution Over Time
ax8 = fig.add_subplot(4, 3, 8)
regime_by_year = pd.crosstab(pivot_df_clean['year'], pivot_df_clean['weather_regime'], normalize='index') * 100
regime_by_year.plot(kind='area', stacked=True, ax=ax8, 
                    color=[regime_colors.get(c, 'gray') for c in regime_by_year.columns],
                    alpha=0.8)
ax8.set_xlabel('Year')
ax8.set_ylabel('Regime Distribution (%)')
ax8.set_title('📈 Regime Evolution Over Time', fontweight='bold')
ax8.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1))

# Plot 9: Decision Priority Matrix
ax9 = fig.add_subplot(4, 3, 9)
priority_data = {'Sector': [], 'High': [], 'Medium': [], 'Critical': []}
for sector, recs in decisions.items():
    priority_data['Sector'].append(sector.replace('_', '\n'))
    priority_data['High'].append(sum(1 for r in recs if r['priority'] == 'HIGH'))
    priority_data['Medium'].append(sum(1 for r in recs if r['priority'] == 'MEDIUM'))
    priority_data['Critical'].append(sum(1 for r in recs if r['priority'] == 'CRITICAL'))

priority_df = pd.DataFrame(priority_data)
x = np.arange(len(priority_df['Sector']))
width = 0.25

ax9.bar(x - width, priority_df['Critical'], width, label='Critical', color=COLORS['warning'])
ax9.bar(x, priority_df['High'], width, label='High', color=COLORS['accent'])
ax9.bar(x + width, priority_df['Medium'], width, label='Medium', color=COLORS['neutral'])

ax9.set_ylabel('Number of Recommendations')
ax9.set_xticks(x)
ax9.set_xticklabels(priority_df['Sector'], fontsize=8)
ax9.set_title('🎯 Decision Priority by Sector', fontweight='bold')
ax9.legend()

# --- Row 4: Key Insights Dashboard ---
# Plot 10: Key Statistics Summary
ax10 = fig.add_subplot(4, 3, 10)
ax10.axis('off')
stats_text = f"""
╔══════════════════════════════════════════════════════════════╗
║                    KEY CLIMATE STATISTICS                     ║
╠══════════════════════════════════════════════════════════════╣
║  📅 Analysis Period: {pivot_df_clean['year'].min()} - {pivot_df_clean['year'].max()}                          ║
║  📊 Total Observations: {len(pivot_df_clean)}                                 ║
║                                                              ║
║  🌡️ TEMPERATURE                                              ║
║     Mean: {pivot_df_clean['temp'].mean():.1f}°C | Range: {pivot_df_clean['temp'].min():.1f} - {pivot_df_clean['temp'].max():.1f}°C            ║
║     Trend: {'+' if trend_results['temp']['slope'] > 0 else ''}{trend_results['temp']['trend_per_decade']:.2f}°C/decade                              ║
║                                                              ║
║  🌧️ RAINFALL                                                 ║
║     Mean: {pivot_df_clean['rain'].mean():.1f}mm | Max: {pivot_df_clean['rain'].max():.1f}mm                     ║
║     Trend: {'+' if trend_results['rain']['slope'] > 0 else ''}{trend_results['rain']['trend_per_decade']:.2f}mm/decade                             ║
║                                                              ║
║  🔍 ANOMALIES DETECTED: {len(anomalies)} ({len(anomalies)/len(pivot_df_clean)*100:.1f}%)                        ║
╚══════════════════════════════════════════════════════════════╝
"""
ax10.text(0.1, 0.5, stats_text, transform=ax10.transAxes, fontsize=10,
          verticalalignment='center', fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9))

# Plot 11: Seasonal Patterns
ax11 = fig.add_subplot(4, 3, 11)
monthly_means = pivot_df_clean.groupby('month')[feature_cols].mean()
monthly_means_norm = (monthly_means - monthly_means.min()) / (monthly_means.max() - monthly_means.min())

angles = np.linspace(0, 2 * np.pi, 12, endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

for param, color in zip(feature_cols, [COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['success']]):
    values = monthly_means_norm[param].tolist()
    values += values[:1]
    ax11.plot(angles, values, 'o-', linewidth=2, label=param, color=color)
    ax11.fill(angles, values, alpha=0.1, color=color)

ax11.set_xticks(angles[:-1])
ax11.set_xticklabels([month_names[m] for m in range(1, 13)])
ax11.set_title('🔄 Seasonal Patterns (Normalized)', fontweight='bold')
ax11.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

# Plot 12: Decision Impact Summary
ax12 = fig.add_subplot(4, 3, 12)
ax12.axis('off')

total_recs = sum(len(recs) for recs in decisions.values())
critical_count = sum(1 for recs in decisions.values() for r in recs if r['priority'] == 'CRITICAL')
high_count = sum(1 for recs in decisions.values() for r in recs if r['priority'] == 'HIGH')

decision_text = f"""
╔══════════════════════════════════════════════════════════════╗
║              DECISION SUPPORT SUMMARY                        ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  📋 Total Recommendations: {total_recs}                              ║
║     🔴 Critical Priority: {critical_count}                                  ║
║     🟠 High Priority: {high_count}                                      ║
║     🟢 Medium Priority: {total_recs - critical_count - high_count}                                   ║
║                                                              ║
║  ⚡ IMMEDIATE ACTIONS REQUIRED:                               ║
"""

# Add top 3 most urgent recommendations
urgent_recs = []
for sector, recs in decisions.items():
    for rec in recs:
        urgent_recs.append((sector, rec))

urgent_recs.sort(key=lambda x: {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2}[x[1]['priority']])

for i, (sector, rec) in enumerate(urgent_recs[:3], 1):
    short_rec = rec['recommendation'][:45] + '...' if len(rec['recommendation']) > 45 else rec['recommendation']
    decision_text += f"║     {i}. [{rec['priority']}] {short_rec}  ║\n"

decision_text += """║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""

ax12.text(0.1, 0.5, decision_text, transform=ax12.transAxes, fontsize=9,
          verticalalignment='center', fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='#fff3cd', alpha=0.9))

plt.tight_layout()
plt.savefig('weather_analysis_dashboard.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("\n   ✓ Dashboard saved as 'weather_analysis_dashboard.png'")

# ============================================================
# STEP 7: EXPORT RESULTS
# ============================================================
print("\n" + "=" * 70)
print("  PHASE 6: EXPORTING RESULTS")
print("=" * 70)

# Export main results
output_df = pivot_df_clean[['year', 'month', 'evap', 'rain', 'soilMoist', 'temp', 
                            'PC1', 'PC2', 'cluster', 'weather_regime',
                            'anomaly_score', 'is_anomaly']].copy()
output_df.to_csv('weather_regime_results.csv', index=False)
print("   ✓ Results saved to 'weather_regime_results.csv'")

# Export trend summary
trend_df = pd.DataFrame(trend_results).T
trend_df.to_csv('trend_analysis_results.csv')
print("   ✓ Trend analysis saved to 'trend_analysis_results.csv'")

# Export anomaly details
anomaly_export = pivot_df_clean[pivot_df_clean['is_anomaly']][
    ['year', 'month', 'evap', 'rain', 'soilMoist', 'temp', 'anomaly_score', 'weather_regime']
].copy()
anomaly_export.to_csv('anomalies_detected.csv', index=False)
print("   ✓ Anomalies saved to 'anomalies_detected.csv'")

# Export decision recommendations
with open('decision_recommendations.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("  CLIMATE-SMART DECISION RECOMMENDATIONS\n")
    f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 70 + "\n\n")
    
    for sector, recs in decisions.items():
        if recs:
            f.write(f"\n{'='*50}\n")
            f.write(f"  {sector.upper().replace('_', ' ')}\n")
            f.write(f"{'='*50}\n\n")
            for i, rec in enumerate(recs, 1):
                f.write(f"  Recommendation {i}:\n")
                f.write(f"  Priority: {rec['priority']}\n")
                f.write(f"  Finding: {rec['finding']}\n")
                f.write(f"  Action: {rec['recommendation']}\n")
                f.write(f"  Timeline: {rec['timeline']}\n\n")

print("   ✓ Recommendations saved to 'decision_recommendations.txt'")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("  ANALYSIS COMPLETE!")
print("=" * 70)
print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    ENHANCED ANALYSIS PIPELINE                        │
├─────────────────────────────────────────────────────────────────────┤
│  ✓ Phase 1: Weather Regime Clustering (5 regimes identified)       │
│  ✓ Phase 2: Trend Analysis (linear regression + Mann-Kendall)      │
│  ✓ Phase 3: Anomaly Detection (Isolation Forest)                   │
│  ✓ Phase 4: Decision Support Recommendations                        │
│  ✓ Phase 5: Comprehensive Dashboard Visualization                   │
│  ✓ Phase 6: Results Export                                          │
├─────────────────────────────────────────────────────────────────────┤
│                         OUTPUT FILES                                 │
├─────────────────────────────────────────────────────────────────────┤
│  📊 weather_analysis_dashboard.png  - Visual dashboard              │
│  📋 weather_regime_results.csv      - All results with anomalies    │
│  📈 trend_analysis_results.csv      - Trend statistics              │
│  🔍 anomalies_detected.csv          - Anomaly details               │
│  📝 decision_recommendations.txt    - Action recommendations        │
└─────────────────────────────────────────────────────────────────────┘
""")

print("\n🎯 KEY TAKEAWAYS FOR DECISION MAKERS:")
print("-" * 50)

# Print key insights
if trend_results['temp']['significant']:
    direction = "warming" if trend_results['temp']['slope'] > 0 else "cooling"
    print(f"   • Significant {direction} trend detected ({trend_results['temp']['trend_per_decade']:.2f}°C/decade)")

if len(anomalies) > 0:
    print(f"   • {len(anomalies)} anomalous months identified requiring attention")
    
highest_anomaly_month = monthly_anomaly_rate.idxmax()
print(f"   • {month_names[highest_anomaly_month]} shows highest anomaly rate ({monthly_anomaly_rate[highest_anomaly_month]:.1f}%)")

print(f"   • {sum(len(r) for r in decisions.values())} actionable recommendations generated")
print("\n" + "=" * 70)
