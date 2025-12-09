"""
Multi-District Weather Analysis with Trend & Anomaly Detection
===============================================================
Analyzes climate data for all districts in Bagmati Province:
- Processes each district individually
- Generates separate results for each district
- Creates comparative cross-district analysis
- Provides decision support recommendations per district

Pipeline: Load All Data → Per-District Analysis → Comparative Analysis → Export
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
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'success': '#2A9D8F',
    'warning': '#E63946',
    'anomaly': '#E63946',
    'normal': '#457B9D',
    'trend_up': '#E07A5F',
    'trend_down': '#3D405B',
    'neutral': '#81B29A'
}

REGIME_COLORS = {
    'A: Hot & Dry': '#FF6B6B', 
    'B: Cold & Dry': '#4ECDC4', 
    'C: Warm & Humid': '#45B7D1', 
    'D: Extreme Rainfall': '#96CEB4',
    'E: Pre-monsoon Storms': '#FFEAA7'
}

MONTH_NAMES = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
               7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def calculate_trend(data, time_index):
    """Calculate linear trend with statistical significance"""
    if len(data) < 3:
        return {'slope': 0, 'r_squared': 0, 'p_value': 1, 'trend_per_decade': 0, 
                'mk_z': 0, 'mk_p_value': 1, 'significant': False}
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(time_index, data)
    
    # Mann-Kendall trend test (simplified)
    n = len(data)
    s = 0
    for i in range(n-1):
        for j in range(i+1, n):
            s += np.sign(data.iloc[j] - data.iloc[i])
    
    var_s = (n * (n - 1) * (2 * n + 5)) / 18
    if var_s > 0:
        if s > 0:
            z_mk = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z_mk = (s + 1) / np.sqrt(var_s)
        else:
            z_mk = 0
        mk_p_value = 2 * (1 - stats.norm.cdf(abs(z_mk)))
    else:
        z_mk = 0
        mk_p_value = 1
    
    return {
        'slope': slope,
        'r_squared': r_value**2,
        'p_value': p_value,
        'trend_per_decade': slope * 10,
        'mk_z': z_mk,
        'mk_p_value': mk_p_value,
        'significant': p_value < 0.05
    }


def assign_weather_regime(cluster_stats, n_clusters=5):
    """Assign weather regime labels based on cluster characteristics"""
    labels = {}
    feature_cols = ['evap', 'rain', 'soilMoist', 'temp']
    
    # Check if we have enough variation
    if cluster_stats[feature_cols].std().sum() == 0:
        for i in range(n_clusters):
            labels[i] = f"Cluster {i}"
        return labels
    
    norm_stats = (cluster_stats - cluster_stats.min()) / (cluster_stats.max() - cluster_stats.min() + 1e-10)
    
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
    for _ in range(min(n_clusters, len(regime_names))):
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
    
    # Handle remaining clusters
    for cluster in cluster_stats.index:
        if cluster not in labels:
            labels[cluster] = f"Cluster {cluster}"
    
    return labels


def generate_decisions(trend_results, monthly_anomaly_rate, regime_distribution, district_name):
    """Generate actionable recommendations based on analysis"""
    decisions = {
        'agriculture': [],
        'water_management': [],
        'urban_planning': [],
        'disaster_preparedness': [],
        'health_sector': []
    }
    
    # Based on trends
    if 'temp' in trend_results and trend_results['temp']['significant'] and trend_results['temp']['slope'] > 0:
        decisions['agriculture'].append({
            'priority': 'HIGH',
            'finding': f"Temperature increasing by {trend_results['temp']['trend_per_decade']:.2f}°C per decade in {district_name}",
            'recommendation': "Shift to heat-tolerant crop varieties; adjust planting calendars",
            'timeline': 'Immediate planning, implement within 1-2 growing seasons'
        })
        decisions['health_sector'].append({
            'priority': 'HIGH',
            'finding': f"Warming trend detected in {district_name}",
            'recommendation': "Expand heat-health warning systems; increase cooling centers",
            'timeline': 'Before next summer season'
        })
    
    if 'rain' in trend_results and trend_results['rain']['significant']:
        if trend_results['rain']['slope'] < 0:
            decisions['water_management'].append({
                'priority': 'CRITICAL',
                'finding': f"Rainfall declining by {abs(trend_results['rain']['trend_per_decade']):.2f}mm per decade in {district_name}",
                'recommendation': "Invest in water storage infrastructure; implement water recycling",
                'timeline': 'Begin infrastructure planning immediately'
            })
            decisions['agriculture'].append({
                'priority': 'HIGH',
                'finding': f"Declining precipitation trend in {district_name}",
                'recommendation': "Adopt drip irrigation; select drought-resistant varieties",
                'timeline': 'Implement in next growing season'
            })
        else:
            decisions['disaster_preparedness'].append({
                'priority': 'HIGH',
                'finding': f"Rainfall increasing by {trend_results['rain']['trend_per_decade']:.2f}mm per decade in {district_name}",
                'recommendation': "Upgrade drainage systems; strengthen flood early warning",
                'timeline': 'Complete upgrades before monsoon'
            })
    
    if 'soilMoist' in trend_results and trend_results['soilMoist']['significant'] and trend_results['soilMoist']['slope'] < 0:
        decisions['agriculture'].append({
            'priority': 'HIGH',
            'finding': f"Declining soil moisture trend in {district_name}",
            'recommendation': "Implement mulching and cover cropping; improve soil organic matter",
            'timeline': 'Start soil improvement programs within 6 months'
        })
    
    # Based on anomaly patterns
    if len(monthly_anomaly_rate) > 0:
        high_anomaly_months = [m for m in range(1, 13) 
                              if m in monthly_anomaly_rate.index and monthly_anomaly_rate[m] > 15]
        
        if high_anomaly_months:
            decisions['disaster_preparedness'].append({
                'priority': 'MEDIUM',
                'finding': f"High anomaly rates in {district_name} during: {[MONTH_NAMES[m] for m in high_anomaly_months]}",
                'recommendation': "Enhanced monitoring during these months; pre-position emergency resources",
                'timeline': 'Update protocols before identified months'
            })
    
    # Based on regime analysis
    if 'D: Extreme Rainfall' in regime_distribution:
        extreme_rain_pct = regime_distribution.get('D: Extreme Rainfall', 0) * 100
        if extreme_rain_pct > 10:
            decisions['urban_planning'].append({
                'priority': 'HIGH',
                'finding': f"Extreme rainfall regime occurs {extreme_rain_pct:.1f}% of the time in {district_name}",
                'recommendation': "Review and expand urban drainage capacity; restrict development in flood zones",
                'timeline': 'Include in next urban development plan'
            })
    
    if 'A: Hot & Dry' in regime_distribution:
        hot_dry_pct = regime_distribution.get('A: Hot & Dry', 0) * 100
        if hot_dry_pct > 15:
            decisions['water_management'].append({
                'priority': 'MEDIUM',
                'finding': f"Hot & dry conditions occur {hot_dry_pct:.1f}% of the time in {district_name}",
                'recommendation': "Develop water rationing protocols; promote rainwater harvesting",
                'timeline': 'Policy ready before dry season'
            })
    
    return decisions


def analyze_single_district(df_district, district_name, output_dir, feature_cols=['evap', 'rain', 'soilMoist', 'temp']):
    """
    Perform complete analysis for a single district.
    Returns: dict with all analysis results
    """
    results = {
        'district': district_name,
        'success': False,
        'error': None
    }
    
    try:
        # Pivot data
        pivot_df = df_district.pivot_table(
            index=['year', 'month'],
            columns='param',
            values='mean',
            aggfunc='first'
        ).reset_index()
        
        # Check if we have all required features
        missing_cols = [col for col in feature_cols if col not in pivot_df.columns]
        if missing_cols:
            results['error'] = f"Missing columns: {missing_cols}"
            return results
        
        pivot_df['date'] = pd.to_datetime(pivot_df['year'].astype(str) + '-' + 
                                          pivot_df['month'].astype(str) + '-01')
        pivot_df['time_index'] = (pivot_df['year'] - pivot_df['year'].min()) * 12 + pivot_df['month']
        
        # Clean data
        X = pivot_df[feature_cols].copy()
        X = X.dropna()
        
        if len(X) < 10:
            results['error'] = f"Insufficient data points: {len(X)}"
            return results
        
        pivot_df_clean = pivot_df.loc[X.index].copy()
        
        # Standardize and PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        n_components = min(2, len(feature_cols))
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Clustering
        n_clusters = min(5, len(X) // 5)  # Ensure enough samples per cluster
        if n_clusters < 2:
            n_clusters = 2
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_pca)
        
        pivot_df_clean['cluster'] = clusters
        pivot_df_clean['PC1'] = X_pca[:, 0]
        if n_components > 1:
            pivot_df_clean['PC2'] = X_pca[:, 1]
        else:
            pivot_df_clean['PC2'] = 0
        
        # Assign weather regimes
        cluster_stats = pivot_df_clean.groupby('cluster')[feature_cols].mean()
        regime_labels = assign_weather_regime(cluster_stats, n_clusters)
        pivot_df_clean['weather_regime'] = pivot_df_clean['cluster'].map(regime_labels)
        
        # Trend Analysis
        yearly_avg = pivot_df_clean.groupby('year')[feature_cols].mean().reset_index()
        trend_results = {}
        for param in feature_cols:
            if param in yearly_avg.columns:
                trend_results[param] = calculate_trend(yearly_avg[param], yearly_avg['year'])
        
        # Anomaly Detection
        iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42,
            max_samples='auto'
        )
        
        anomaly_labels = iso_forest.fit_predict(X_scaled)
        anomaly_scores = iso_forest.decision_function(X_scaled)
        
        pivot_df_clean['anomaly_label'] = anomaly_labels
        pivot_df_clean['anomaly_score'] = anomaly_scores
        pivot_df_clean['is_anomaly'] = anomaly_labels == -1
        
        # Monthly anomaly rates
        monthly_anomaly_rate = pivot_df_clean.groupby('month')['is_anomaly'].mean() * 100
        
        # Regime distribution
        regime_distribution = pivot_df_clean['weather_regime'].value_counts(normalize=True).to_dict()
        
        # Generate decisions
        decisions = generate_decisions(trend_results, monthly_anomaly_rate, regime_distribution, district_name)
        
        # Store results
        results['success'] = True
        results['data'] = pivot_df_clean
        results['trend_results'] = trend_results
        results['anomaly_count'] = pivot_df_clean['is_anomaly'].sum()
        results['total_points'] = len(pivot_df_clean)
        results['anomaly_rate'] = results['anomaly_count'] / results['total_points'] * 100
        results['monthly_anomaly_rate'] = monthly_anomaly_rate
        results['regime_distribution'] = regime_distribution
        results['cluster_stats'] = cluster_stats
        results['regime_labels'] = regime_labels
        results['decisions'] = decisions
        results['pca'] = pca
        results['scaler'] = scaler
        results['year_range'] = (pivot_df_clean['year'].min(), pivot_df_clean['year'].max())
        
        # Export district-specific files
        os.makedirs(output_dir, exist_ok=True)
        
        # Save data
        pivot_df_clean.to_csv(os.path.join(output_dir, f'{district_name}_results.csv'), index=False)
        
        # Save trends
        trend_df = pd.DataFrame(trend_results).T
        trend_df.to_csv(os.path.join(output_dir, f'{district_name}_trends.csv'))
        
        # Save anomalies
        anomalies = pivot_df_clean[pivot_df_clean['is_anomaly']]
        if len(anomalies) > 0:
            anomalies.to_csv(os.path.join(output_dir, f'{district_name}_anomalies.csv'), index=False)
        
        # Save recommendations
        with open(os.path.join(output_dir, f'{district_name}_recommendations.txt'), 'w') as f:
            f.write(f"{'='*60}\n")
            f.write(f"  CLIMATE RECOMMENDATIONS FOR {district_name.upper()}\n")
            f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n\n")
            
            for sector, recs in decisions.items():
                if recs:
                    f.write(f"\n{'-'*50}\n")
                    f.write(f"  {sector.upper().replace('_', ' ')}\n")
                    f.write(f"{'-'*50}\n\n")
                    for i, rec in enumerate(recs, 1):
                        f.write(f"  [{rec['priority']}] {rec['finding']}\n")
                        f.write(f"  → Action: {rec['recommendation']}\n")
                        f.write(f"  → Timeline: {rec['timeline']}\n\n")
        
        # Create district visualization
        create_district_dashboard(results, output_dir, district_name)
        
    except Exception as e:
        results['error'] = str(e)
        
    return results


def create_district_dashboard(results, output_dir, district_name):
    """Create a dashboard for a single district"""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Climate Analysis Dashboard: {district_name}\nTrend & Anomaly Detection', 
                 fontsize=14, fontweight='bold')
    
    data = results['data']
    trend_results = results['trend_results']
    feature_cols = ['evap', 'rain', 'soilMoist', 'temp']
    
    # Plot 1: Temperature Trend
    ax1 = fig.add_subplot(2, 3, 1)
    yearly_avg = data.groupby('year')[feature_cols].mean().reset_index()
    if 'temp' in yearly_avg.columns and len(yearly_avg) > 1:
        ax1.scatter(yearly_avg['year'], yearly_avg['temp'], color=COLORS['primary'], alpha=0.7, s=50)
        z = np.polyfit(yearly_avg['year'], yearly_avg['temp'], 1)
        p = np.poly1d(z)
        ax1.plot(yearly_avg['year'], p(yearly_avg['year']), color=COLORS['trend_up'], linewidth=2)
        ax1.set_title('🌡️ Temperature Trend', fontweight='bold')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Temperature (°C)')
        if 'temp' in trend_results:
            sig = "✓" if trend_results['temp']['significant'] else "○"
            ax1.text(0.98, 0.02, f"{sig} {trend_results['temp']['trend_per_decade']:.2f}°C/decade", 
                    transform=ax1.transAxes, ha='right', fontsize=9)
    
    # Plot 2: Rainfall Trend
    ax2 = fig.add_subplot(2, 3, 2)
    if 'rain' in yearly_avg.columns and len(yearly_avg) > 1:
        ax2.scatter(yearly_avg['year'], yearly_avg['rain'], color=COLORS['secondary'], alpha=0.7, s=50)
        z = np.polyfit(yearly_avg['year'], yearly_avg['rain'], 1)
        p = np.poly1d(z)
        ax2.plot(yearly_avg['year'], p(yearly_avg['year']), color=COLORS['trend_down'], linewidth=2)
        ax2.set_title('🌧️ Rainfall Trend', fontweight='bold')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Rainfall (mm)')
        if 'rain' in trend_results:
            sig = "✓" if trend_results['rain']['significant'] else "○"
            ax2.text(0.98, 0.02, f"{sig} {trend_results['rain']['trend_per_decade']:.2f}mm/decade", 
                    transform=ax2.transAxes, ha='right', fontsize=9)
    
    # Plot 3: Anomaly Timeline
    ax3 = fig.add_subplot(2, 3, 3)
    normal_data = data[~data['is_anomaly']]
    anomaly_data = data[data['is_anomaly']]
    ax3.scatter(normal_data['date'], normal_data['anomaly_score'], c=COLORS['normal'], 
                alpha=0.5, s=20, label='Normal')
    if len(anomaly_data) > 0:
        ax3.scatter(anomaly_data['date'], anomaly_data['anomaly_score'], c=COLORS['anomaly'], 
                    alpha=0.8, s=50, marker='X', label='Anomaly')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_title('🔍 Anomaly Timeline', fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Anomaly Score')
    ax3.legend(fontsize=8)
    
    # Plot 4: Monthly Anomaly Rates
    ax4 = fig.add_subplot(2, 3, 4)
    monthly_rate = results['monthly_anomaly_rate']
    colors_month = [COLORS['anomaly'] if r > 10 else COLORS['normal'] for r in monthly_rate]
    ax4.bar([MONTH_NAMES[m] for m in monthly_rate.index], monthly_rate.values, color=colors_month, alpha=0.8)
    ax4.axhline(y=10, color=COLORS['warning'], linestyle='--', alpha=0.7)
    ax4.set_title('📅 Monthly Anomaly Rates', fontweight='bold')
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Anomaly Rate (%)')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 5: Weather Regimes in PCA Space
    ax5 = fig.add_subplot(2, 3, 5)
    for regime in data['weather_regime'].unique():
        mask = data['weather_regime'] == regime
        color = REGIME_COLORS.get(regime, 'gray')
        ax5.scatter(data.loc[mask, 'PC1'], data.loc[mask, 'PC2'],
                    label=regime, alpha=0.6, c=color, s=40)
    ax5.set_title('🌐 Weather Regimes (PCA)', fontweight='bold')
    ax5.set_xlabel('PC1')
    ax5.set_ylabel('PC2')
    ax5.legend(fontsize=7, loc='upper right')
    
    # Plot 6: Summary Statistics
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"""
╔══════════════════════════════════════════╗
║         {district_name.upper():^30} ║
╠══════════════════════════════════════════╣
║  Period: {results['year_range'][0]} - {results['year_range'][1]}                    ║
║  Data Points: {results['total_points']}                        ║
║  Anomalies: {results['anomaly_count']} ({results['anomaly_rate']:.1f}%)                  ║
╠══════════════════════════════════════════╣
║  SIGNIFICANT TRENDS:                     ║"""
    
    for param, trend in trend_results.items():
        if trend['significant']:
            direction = "↑" if trend['slope'] > 0 else "↓"
            summary_text += f"\n║    {param}: {direction} {abs(trend['trend_per_decade']):.2f}/decade     ║"
    
    summary_text += """
╚══════════════════════════════════════════╝
"""
    
    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{district_name}_dashboard.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_comparative_dashboard(all_results, output_dir):
    """Create a comparative dashboard across all districts"""
    
    # Filter successful results
    successful = {k: v for k, v in all_results.items() if v['success']}
    
    if len(successful) == 0:
        print("   ⚠️ No successful district analyses to compare")
        return
    
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Comparative Climate Analysis: All Districts\nBagmati Province Overview', 
                 fontsize=16, fontweight='bold')
    
    districts = list(successful.keys())
    
    # Plot 1: Temperature Trends Comparison
    ax1 = fig.add_subplot(2, 3, 1)
    temp_trends = []
    for d in districts:
        if 'temp' in successful[d]['trend_results']:
            temp_trends.append(successful[d]['trend_results']['temp']['trend_per_decade'])
        else:
            temp_trends.append(0)
    
    colors = [COLORS['trend_up'] if t > 0 else COLORS['trend_down'] for t in temp_trends]
    bars = ax1.barh(districts, temp_trends, color=colors, alpha=0.8)
    ax1.axvline(x=0, color='gray', linestyle='--')
    ax1.set_xlabel('Temperature Change (°C/decade)')
    ax1.set_title('🌡️ Temperature Trends by District', fontweight='bold')
    
    # Add significance markers
    for i, d in enumerate(districts):
        if 'temp' in successful[d]['trend_results'] and successful[d]['trend_results']['temp']['significant']:
            ax1.text(temp_trends[i], i, ' *', va='center', fontsize=12, fontweight='bold')
    
    # Plot 2: Rainfall Trends Comparison
    ax2 = fig.add_subplot(2, 3, 2)
    rain_trends = []
    for d in districts:
        if 'rain' in successful[d]['trend_results']:
            rain_trends.append(successful[d]['trend_results']['rain']['trend_per_decade'])
        else:
            rain_trends.append(0)
    
    colors = [COLORS['trend_up'] if t > 0 else COLORS['trend_down'] for t in rain_trends]
    bars = ax2.barh(districts, rain_trends, color=colors, alpha=0.8)
    ax2.axvline(x=0, color='gray', linestyle='--')
    ax2.set_xlabel('Rainfall Change (mm/decade)')
    ax2.set_title('🌧️ Rainfall Trends by District', fontweight='bold')
    
    for i, d in enumerate(districts):
        if 'rain' in successful[d]['trend_results'] and successful[d]['trend_results']['rain']['significant']:
            ax2.text(rain_trends[i], i, ' *', va='center', fontsize=12, fontweight='bold')
    
    # Plot 3: Anomaly Rates Comparison
    ax3 = fig.add_subplot(2, 3, 3)
    anomaly_rates = [successful[d]['anomaly_rate'] for d in districts]
    colors = [COLORS['anomaly'] if r > 10 else COLORS['normal'] for r in anomaly_rates]
    ax3.barh(districts, anomaly_rates, color=colors, alpha=0.8)
    ax3.axvline(x=10, color=COLORS['warning'], linestyle='--', label='10% threshold')
    ax3.set_xlabel('Anomaly Rate (%)')
    ax3.set_title('🔍 Anomaly Rates by District', fontweight='bold')
    ax3.legend(fontsize=8)
    
    # Plot 4: Regime Distribution Heatmap
    ax4 = fig.add_subplot(2, 3, 4)
    regime_data = []
    all_regimes = set()
    for d in districts:
        all_regimes.update(successful[d]['regime_distribution'].keys())
    all_regimes = sorted(list(all_regimes))
    
    for d in districts:
        row = [successful[d]['regime_distribution'].get(r, 0) * 100 for r in all_regimes]
        regime_data.append(row)
    
    regime_df = pd.DataFrame(regime_data, index=districts, columns=all_regimes)
    sns.heatmap(regime_df, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax4, cbar_kws={'label': '%'})
    ax4.set_title('🌐 Weather Regime Distribution (%)', fontweight='bold')
    ax4.set_xlabel('Weather Regime')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 5: Priority Recommendations Count
    ax5 = fig.add_subplot(2, 3, 5)
    priority_counts = {'CRITICAL': [], 'HIGH': [], 'MEDIUM': []}
    for d in districts:
        counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0}
        for sector_recs in successful[d]['decisions'].values():
            for rec in sector_recs:
                if rec['priority'] in counts:
                    counts[rec['priority']] += 1
        for p in priority_counts:
            priority_counts[p].append(counts[p])
    
    x = np.arange(len(districts))
    width = 0.25
    ax5.bar(x - width, priority_counts['CRITICAL'], width, label='Critical', color=COLORS['warning'])
    ax5.bar(x, priority_counts['HIGH'], width, label='High', color=COLORS['accent'])
    ax5.bar(x + width, priority_counts['MEDIUM'], width, label='Medium', color=COLORS['neutral'])
    ax5.set_xticks(x)
    ax5.set_xticklabels(districts, rotation=45, ha='right')
    ax5.set_ylabel('Number of Recommendations')
    ax5.set_title('🎯 Recommendations Priority by District', fontweight='bold')
    ax5.legend()
    
    # Plot 6: Summary Table
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create summary table
    summary_data = []
    for d in districts:
        r = successful[d]
        temp_trend = r['trend_results'].get('temp', {}).get('trend_per_decade', 0)
        rain_trend = r['trend_results'].get('rain', {}).get('trend_per_decade', 0)
        summary_data.append({
            'District': d,
            'Years': f"{r['year_range'][0]}-{r['year_range'][1]}",
            'Points': r['total_points'],
            'Anomalies': f"{r['anomaly_rate']:.1f}%",
            'Temp Trend': f"{temp_trend:+.2f}°C",
            'Rain Trend': f"{rain_trend:+.2f}mm"
        })
    
    summary_df = pd.DataFrame(summary_data)
    table = ax6.table(cellText=summary_df.values, colLabels=summary_df.columns,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax6.set_title('📊 District Summary Table', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparative_dashboard.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✓ Comparative dashboard saved")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("=" * 70)
    print("  MULTI-DISTRICT CLIMATE ANALYSIS")
    print("  Bagmati Province - Trend & Anomaly Detection")
    print("=" * 70)
    
    # Load data
    data_path = "data/second/data_all_province.csv"
    districts_path = "data/second/districts.csv"
    
    print("\n📂 Loading data...")
    df = pd.read_csv(data_path)
    districts_df = pd.read_csv(districts_path)
    districts = districts_df['district'].tolist()
    
    print(f"   Total records: {len(df)}")
    print(f"   Districts to analyze: {len(districts)}")
    print(f"   Districts: {', '.join(districts)}")
    
    # Create output directory structure
    base_output_dir = "district_results"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Analyze each district
    print("\n" + "=" * 70)
    print("  ANALYZING INDIVIDUAL DISTRICTS")
    print("=" * 70)
    
    all_results = {}
    
    for i, district in enumerate(districts, 1):
        print(f"\n[{i}/{len(districts)}] Processing {district}...")
        
        df_district = df[df['district'] == district].copy()
        
        if len(df_district) == 0:
            print(f"   ⚠️ No data found for {district}")
            continue
        
        output_dir = os.path.join(base_output_dir, district.replace(' ', '_'))
        results = analyze_single_district(df_district, district, output_dir)
        all_results[district] = results
        
        if results['success']:
            print(f"   ✓ {district}: {results['total_points']} points, "
                  f"{results['anomaly_count']} anomalies ({results['anomaly_rate']:.1f}%)")
            
            # Print significant trends
            sig_trends = [p for p, t in results['trend_results'].items() if t['significant']]
            if sig_trends:
                print(f"   📈 Significant trends: {', '.join(sig_trends)}")
        else:
            print(f"   ⚠️ {district}: {results['error']}")
    
    # Create comparative analysis
    print("\n" + "=" * 70)
    print("  CREATING COMPARATIVE ANALYSIS")
    print("=" * 70)
    
    create_comparative_dashboard(all_results, base_output_dir)
    
    # Create master summary CSV
    print("\n   Creating master summary...")
    summary_rows = []
    for district, results in all_results.items():
        if results['success']:
            row = {
                'district': district,
                'year_start': results['year_range'][0],
                'year_end': results['year_range'][1],
                'data_points': results['total_points'],
                'anomaly_count': results['anomaly_count'],
                'anomaly_rate': results['anomaly_rate'],
            }
            for param in ['evap', 'rain', 'soilMoist', 'temp']:
                if param in results['trend_results']:
                    row[f'{param}_trend_per_decade'] = results['trend_results'][param]['trend_per_decade']
                    row[f'{param}_significant'] = results['trend_results'][param]['significant']
            
            # Count recommendations by priority
            for priority in ['CRITICAL', 'HIGH', 'MEDIUM']:
                count = sum(1 for sector_recs in results['decisions'].values() 
                           for rec in sector_recs if rec['priority'] == priority)
                row[f'recommendations_{priority.lower()}'] = count
            
            summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(base_output_dir, 'all_districts_summary.csv'), index=False)
    print(f"   ✓ Summary saved to {base_output_dir}/all_districts_summary.csv")
    
    # Create master recommendations file
    print("\n   Creating master recommendations...")
    with open(os.path.join(base_output_dir, 'all_districts_recommendations.txt'), 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("  CLIMATE-SMART RECOMMENDATIONS - ALL DISTRICTS\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n")
        
        for district, results in all_results.items():
            if results['success']:
                has_recs = any(len(recs) > 0 for recs in results['decisions'].values())
                if has_recs:
                    f.write(f"\n\n{'='*60}\n")
                    f.write(f"  {district.upper()}\n")
                    f.write(f"{'='*60}\n")
                    
                    for sector, recs in results['decisions'].items():
                        if recs:
                            f.write(f"\n  {sector.upper().replace('_', ' ')}:\n")
                            for rec in recs:
                                f.write(f"    [{rec['priority']}] {rec['finding']}\n")
                                f.write(f"    → {rec['recommendation']}\n\n")
    
    print(f"   ✓ Recommendations saved to {base_output_dir}/all_districts_recommendations.txt")
    
    # Final summary
    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE!")
    print("=" * 70)
    
    successful = sum(1 for r in all_results.values() if r['success'])
    failed = sum(1 for r in all_results.values() if not r['success'])
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                    MULTI-DISTRICT ANALYSIS SUMMARY                   │
├─────────────────────────────────────────────────────────────────────┤
│  Districts Analyzed: {successful}/{len(districts)} successful                            │
│  Failed/Skipped: {failed}                                                │
├─────────────────────────────────────────────────────────────────────┤
│                         OUTPUT STRUCTURE                             │
├─────────────────────────────────────────────────────────────────────┤
│  📁 district_results/                                                │
│     ├── comparative_dashboard.png     (cross-district comparison)   │
│     ├── all_districts_summary.csv     (master summary data)         │
│     ├── all_districts_recommendations.txt                           │
│     │                                                               │
│     ├── 📁 [District_Name]/                                          │
│     │   ├── [District]_dashboard.png                                │
│     │   ├── [District]_results.csv                                  │
│     │   ├── [District]_trends.csv                                   │
│     │   ├── [District]_anomalies.csv                                │
│     │   └── [District]_recommendations.txt                          │
│     └── ... (one folder per district)                               │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    # Print key findings
    print("\n🎯 KEY FINDINGS ACROSS DISTRICTS:")
    print("-" * 50)
    
    # Find districts with most significant trends
    sig_temp_districts = [d for d, r in all_results.items() 
                         if r['success'] and 'temp' in r['trend_results'] 
                         and r['trend_results']['temp']['significant']]
    if sig_temp_districts:
        print(f"   🌡️ Significant warming: {', '.join(sig_temp_districts)}")
    
    sig_rain_districts = [d for d, r in all_results.items() 
                         if r['success'] and 'rain' in r['trend_results'] 
                         and r['trend_results']['rain']['significant']]
    if sig_rain_districts:
        print(f"   🌧️ Significant rainfall change: {', '.join(sig_rain_districts)}")
    
    # Districts with highest anomaly rates
    anomaly_sorted = sorted(
        [(d, r['anomaly_rate']) for d, r in all_results.items() if r['success']],
        key=lambda x: x[1], reverse=True
    )[:3]
    print(f"   🔍 Highest anomaly rates:")
    for d, rate in anomaly_sorted:
        print(f"      • {d}: {rate:.1f}%")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

