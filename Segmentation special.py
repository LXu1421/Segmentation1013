import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from skimage import feature
import matplotlib.pyplot as plt

# Import data
CHM1 = pd.read_csv('CHM1.csv').values
UTME1 = pd.read_csv('UTME1.csv').values
UTMN1 = pd.read_csv('UTMN1.csv').values
TreeMask1 = pd.read_csv('TreeMask1.csv').values

Data = CHM1
threshold = 0.5

# Preallocate memory for the output variables
FinalSegmentedRegions = np.zeros_like(Data, dtype=bool)
SmoothingFactorMap = np.zeros_like(Data)
scaleValues = np.arange(2.55, 3.45, 0.05)
numScales = len(scaleValues)
edgR = np.zeros((numScales, 4))  # 4th column for straight border ratio

# Define loop range and pre-calculate values
edgR_F = 0
for i in scaleValues:
    smoothedData = gaussian_filter(Data, i)  # Gaussian smoothing
    invertedData = -smoothedData  # Invert the data
    L = watershed(invertedData, mask=TreeMask1)  # Watershed segmentation
    L[~TreeMask1] = 0  # Apply mask to exclude non-tree regions

    edgR[edgR_F, 0] = i  # Store current smoothing factor

    # Get unique labels excluding the background
    uniqueLabels = np.unique(L)
    uniqueLabels = uniqueLabels[uniqueLabels != 0]

    for region_label in uniqueLabels:  # Rename loop variable to avoid conflict
        regionMask = (L == region_label)  # Mask for the current region
        CC = label(regionMask)  # Get connected components

        # Get region properties
        regionProps = regionprops(CC)

        for prop in regionProps:
            if prop.area > 1:
                # Check for straight edges in the region
                edgeMap = feature.canny(regionMask, sigma=1)

                # Count vertical and horizontal edges
                horizontalEdges = np.sum(edgeMap[1:, :] & edgeMap[:-1, :])
                verticalEdges = np.sum(edgeMap[:, 1:] & edgeMap[:, :-1])

                # Ratio of straight edges
                totalEdges = np.sum(edgeMap)
                if totalEdges > 0:
                    straightEdgeRatio = (horizontalEdges + verticalEdges) / totalEdges
                    edgR[edgR_F, 1] += 1  # Count of processed regions
                    edgR[edgR_F, 2] += straightEdgeRatio  # Sum of straight edge ratios

    edgR_F += 1  # Move to the next scale factor

# Calculate the final straight border ratio
edgR[:, 3] = edgR[:, 2] / edgR[:, 1]

# Plot the results
plt.plot(scaleValues, edgR[:, 3])
plt.title('Final Segmented Regions with Low Straight Border Ratio')
plt.xlabel('Smoothing Factor')
plt.ylabel('Straight Border Ratio')
plt.grid(True)
plt.show()
plt.savefig('Straight Border Ratio.png')
plt.close()


# Find the best smoothing factor
best_i = edgR[np.argmin(edgR[:, 3]), 0]

smoothedData = gaussian_filter(Data, best_i)  # Gaussian smoothing
invertedData = -smoothedData  # Invert the data
L = watershed(invertedData, mask=TreeMask1)  # Watershed segmentation
L[~TreeMask1] = 0  # Apply mask to exclude non-tree regions

# Extract coordinates and CHM values for segmented regions
mask = (L > 0) & (CHM1 > 0)
B = np.zeros((np.sum(mask), 5))
B[:, 0] = UTME1[mask]  # Easting
B[:, 1] = UTMN1[mask]  # Northing
B[:, 2] = L[mask]  # ID
B[:, 3] = CHM1[mask]  # CHM at location

# Calculate the median height of each tree
unique_ids = np.unique(B[:, 2])
for uid in unique_ids:
    B[B[:, 2] == uid, 4] = np.median(B[B[:, 2] == uid, 3])  # Median height of the tree

# Export the results
output_filename = f"Factor{best_i:.2f}_Export.csv"
pd.DataFrame(B).to_csv(output_filename, index=False, header=["Easting", "Northing", "ID", "CHM", "Median Height"])
