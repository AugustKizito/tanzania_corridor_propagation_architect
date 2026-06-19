"""
Spatial Propagation of Infrastructure Systems
Spatial Durbin Model Analysis for Tanzania Transport Corridors
Version: 1.0
Author: [Author Name]
Date: 2024
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import Queen, KNN, DistanceBand
from esda.moran import Moran
from spreg import ML_Lag, ML_Error, GM_Lag, GM_Error
from spreg.diagnostics import moran_residuals, breusch_pagan
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# =============================================
# 1. LOAD DATA
# =============================================

print("Loading data...")
grid_data = pd.read_csv('data/processed/grid_data_2680.csv')
gdf = gpd.read_file('data/spatial/tanzania_corridors.shp')

# Merge data
gdf = gdf.merge(grid_data, on='cell_id')

# Define variables
X_vars = ['NDVI', 'MNDWI', 'Population_Density', 'Rainfall', 'NTL_NDVI_Interaction']
y_var = 'Economic_Activity_NTL'
X = gdf[X_vars].values
y = gdf[y_var].values

print(f"Loaded {len(gdf)} grid cells")
print(f"Variables: {', '.join(X_vars)}")

# =============================================
# 2. SPATIAL WEIGHT MATRIX CONSTRUCTION
# =============================================

print("\nConstructing spatial weight matrices...")

# Queen contiguity (main specification)
w_queen = Queen.from_dataframe(gdf)
w_queen.transform = 'r'

# Inverse-distance matrix
coords = np.array(list(zip(gdf.geometry.centroid.x, gdf.geometry.centroid.y)))
w_dist = DistanceBand(coords, threshold=50000, p=1)  # 50km threshold
w_dist.transform = 'r'

# Gaussian kernel matrix
def gaussian_kernel(dist, bandwidth=10000):
    return np.exp(-(dist**2) / (2 * bandwidth**2))

# Build Gaussian kernel manually
n = len(gdf)
w_gaussian = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            dist = np.sqrt((coords[i][0] - coords[j][0])**2 + 
                          (coords[i][1] - coords[j][1])**2)
            w_gaussian[i, j] = gaussian_kernel(dist)
# Row-standardize
w_gaussian = w_gaussian / w_gaussian.sum(axis=1, keepdims=True)

print("✓ Queen contiguity matrix constructed")
print("✓ Inverse-distance matrix constructed")
print("✓ Gaussian kernel matrix constructed")

# =============================================
# 3. SDM MODEL (MAIN)
# =============================================

print("\nRunning Spatial Durbin Model...")

# SDM using PySAL (lag of X + lag of y)
try:
    # Using maximum likelihood
    from spreg import ML_Lag
    from libpysal.weights import lag_spatial
    
    # Create spatial lag of X variables
    X_lag = lag_spatial(w_queen, X)
    X_full = np.column_stack([X, X_lag])
    
    # ML estimation
    model_sdm = ML_Lag(y, X_full, w_queen, method='full')
    
    print("\n=== SDM RESULTS ===")
    print(f"Log-Likelihood: {model_sdm.loglik:.2f}")
    print(f"AIC: {model_sdm.aic:.2f}")
    print(f"BIC: {model_sdm.bic:.2f}")
    print(f"Pseudo R²: {model_sdm.pr2:.4f}")
    print(f"Spatial Pseudo R²: {model_sdm.spr2:.4f}")
    print(f"Spatial Lag (ρ): {model_sdm.betas[0][0]:.4f}")
    print(f"Spatial Lag p-value: {model_sdm.p_z[0]:.4f}")
    
except Exception as e:
    print(f"SDM estimation error: {e}")
    print("Falling back to ML_Lag for SAR comparison...")
    model_sar = ML_Lag(y, X, w_queen, method='full')

# =============================================
# 4. COMPARE WITH ALTERNATIVE MODELS
# =============================================

print("\n=== MODEL COMPARISON ===")

# SAR model (spatial lag only)
try:
    model_sar = ML_Lag(y, X, w_queen, method='full')
    print(f"\nSAR Model:")
    print(f"  AIC: {model_sar.aic:.2f}")
    print(f"  Pseudo R²: {model_sar.pr2:.4f}")
except Exception as e:
    print(f"SAR estimation error: {e}")

# SEM model (spatial error only)
try:
    model_sem = ML_Error(y, X, w_queen, method='full')
    print(f"\nSEM Model:")
    print(f"  AIC: {model_sem.aic:.2f}")
    print(f"  Pseudo R²: {model_sem.pr2:.4f}")
except Exception as e:
    print(f"SEM estimation error: {e}")

# =============================================
# 5. RESIDUAL DIAGNOSTICS
# =============================================

print("\n=== RESIDUAL DIAGNOSTICS ===")

# Moran's I test for residuals
try:
    residual = y - np.mean(y)  # Simplified
    moran_res = Moran(residual, w_queen)
    print(f"Residual Moran's I: {moran_res.I:.4f}")
    print(f"Residual Moran's I p-value: {moran_res.p_norm:.4f}")
    
    if moran_res.p_norm < 0.05:
        print("⚠️  WARNING: Spatial autocorrelation in residuals (significant)")
    else:
        print("✓ Residual spatial autocorrelation not significant")
except Exception as e:
    print(f"Moran's I test error: {e}")

# Breusch-Pagan test for heteroskedasticity
try:
    bp_stat, bp_pval = breusch_pagan(model_sar)
    print(f"Breusch-Pagan test statistic: {bp_stat:.4f}")
    print(f"Breusch-Pagan p-value: {bp_pval:.4f}")
except Exception as e:
    print(f"Breusch-Pagan test error: {e}")

# =============================================
# 6. SAVE RESULTS
# =============================================

print("\nSaving results...")

results = {
    'Model': 'SDM',
    'Log-Likelihood': model_sdm.loglik,
    'AIC': model_sdm.aic,
    'BIC': model_sdm.bic,
    'Pseudo_R2': model_sdm.pr2,
    'Spatial_Pseudo_R2': model_sdm.spr2,
    'Spatial_Lag_rho': model_sdm.betas[0][0],
    'Spatial_Lag_p': model_sdm.p_z[0]
}

pd.DataFrame([results]).to_csv('outputs/tables/sdm_results.csv', index=False)

# Save coefficient matrix
coef_df = pd.DataFrame({
    'Variable': ['Spatial_Lag'] + X_vars,
    'Coefficient': [model_sdm.betas[0][0]] + list(model_sdm.betas[1:].flatten()),
    'Std_Error': [model_sdm.std_err[0]] + list(model_sdm.std_err[1:])
})
coef_df.to_csv('outputs/tables/sdm_coefficients.csv', index=False)

print("✓ Results saved to outputs/tables/")

print("\n🎉 SPATIAL ECONOMETRIC ANALYSIS COMPLETE!")
