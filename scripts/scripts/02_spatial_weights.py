"""
Spatial Propagation of Infrastructure Systems
Spatial Weight Matrix Construction
Version: 1.0
Author: Kizito Ngowi
Date: 2024
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import Queen, DistanceBand
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("SPATIAL WEIGHT MATRIX CONSTRUCTION")
print("=" * 60)

# =============================================
# 1. LOAD DATA
# =============================================

print("\n[1] Loading data...")

gdf = gpd.read_file('data/spatial/tanzania_grid_cells.shp')
print(f"✓ Loaded {len(gdf)} grid cells")

# Get coordinates
coords = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))

# =============================================
# 2. QUEEN CONTIGUITY MATRIX
# =============================================

print("\n[2] Constructing Queen contiguity matrix...")

w_queen = Queen.from_dataframe(gdf)
w_queen.transform = 'r'  # Row-standardize
print(f"✓ Queen contiguity: {w_queen.n} observations, {w_queen.pct_nonzero:.2f}% nonzero")

# =============================================
# 3. INVERSE-DISTANCE MATRIX
# =============================================

print("\n[3] Constructing inverse-distance matrix...")

# Using 50km threshold (matching manuscript)
w_dist = DistanceBand(coords, threshold=50000, p=1)  # p=1 for inverse distance
w_dist.transform = 'r'
print(f"✓ Inverse-distance: {w_dist.n} observations, {w_dist.pct_nonzero:.2f}% nonzero")

# =============================================
# 4. GAUSSIAN KERNEL MATRIX
# =============================================

print("\n[4] Constructing Gaussian kernel matrix...")

def gaussian_kernel_matrix(coords, bandwidth=10000):
    """Construct Gaussian kernel spatial weight matrix"""
    n = len(coords)
    W = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = np.sqrt((coords[i][0] - coords[j][0])**2 + 
                              (coords[i][1] - coords[j][1])**2)
                W[i, j] = np.exp(-(dist**2) / (2 * bandwidth**2))
    
    # Row-standardize
    row_sums = W.sum(axis=1, keepdims=True)
    W = W / row_sums
    return W

W_gaussian = gaussian_kernel_matrix(coords)
print(f"✓ Gaussian kernel constructed: {W_gaussian.shape}")

# =============================================
# 5. SAVE MATRICES
# =============================================

print("\n[5] Saving spatial weight matrices...")

# Save as numpy arrays
np.save('data/spatial/w_queen.npy', w_queen.full()[0])
np.save('data/spatial/w_distance.npy', w_dist.full()[0])
np.save('data/spatial/w_gaussian.npy', W_gaussian)

print("✓ Saved weight matrices to data/spatial/")

# Save summary
summary = pd.DataFrame({
    'Matrix': ['Queen Contiguity', 'Inverse Distance', 'Gaussian Kernel'],
    'Nonzero (%)': [w_queen.pct_nonzero, w_dist.pct_nonzero, 
                    np.sum(W_gaussian > 0) / (W_gaussian.shape[0]**2) * 100]
})
summary.to_csv('data/spatial/weight_summary.csv', index=False)

print("\n" + "=" * 60)
print("SPATIAL WEIGHT CONSTRUCTION COMPLETE!")
print("=" * 60)
