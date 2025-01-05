#pip install git+https://github.com/facebookresearch/segment-anything.git
#conda install shapely
#conda install geopandas

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import rasterio
from shapely.geometry import Polygon
import geopandas as gpd
from rasterio import features
from rasterio.transform import rowcol
from affine import Affine

# prepare function to plot masks over original image
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    



###############################################
# load geotiff-file and and apply SAM with custom settings
###############################################


# Load geocoded raster (GeoTIFF)
raster_path = 'E:/8_SAM_tutorial/SAM_tutorial.tif'
with rasterio.open(raster_path) as src:
    transform = src.transform
    crs = src.crs
    image = src.read(1)  # Read first band for visualization (if needed)
    rgb_image = np.dstack([src.read(i+1) for i in range(src.count)])  # RGB composite

rgb_image = rgb_image[:, :, [2, 1, 0]]

plt.figure(figsize=(20, 20))
plt.imshow(rgb_image)
plt.axis('off')
plt.show()

# Normalize raster to uint8 if needed
if rgb_image.dtype == np.uint16:
    rgb_image = cv2.normalize(rgb_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


###############################################
# apply SAM with custom settings to geotiff file
###############################################

# Apply SAM to this image
import sys
sys.path.append("E:/Python_environments/sam_checkpoint/")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

sam_checkpoint = "E:/Python_environments/sam_checkpoint/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

# prepare model for application
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)


###############################################
# adjust settings of SAM and apply it to image
###############################################

mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=20,  # Requires open-cv to run post-processing
)


masks2 = mask_generator_2.generate(rgb_image)

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks2)
plt.axis('off')
plt.show() 


# Initialize list to store polygons
polygons = []

from rasterio.transform import xy

# Transform mask contours from pixel to geospatial coordinates
for mask in masks2:
    segmentation = mask['segmentation']

    if np.any(segmentation):
        # Convert mask to uint8
        mask_uint8 = (segmentation * 255).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if len(contour) > 2:
                # Create polygon from contour in pixel coordinates
                polygon = Polygon(contour.reshape(-1, 2))

                # Convert pixel coordinates to geographic coordinates
                geo_coords = [
                    xy(transform, int(col), int(row))  # Swap col and row to match geographic system
                    for row, col in polygon.exterior.coords
                ]
                geo_polygon = Polygon(geo_coords)
                
                polygons.append(geo_polygon)




# Create GeoDataFrame with CRS from the raster
gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)

# Save to GeoPackage or Shapefile
output_path = 'E:/8_SAM_tutorial/'
gdf.to_file(f"{output_path}/output_geocoded1.gpkg", driver="GPKG")
gdf.to_file(f"{output_path}/output_geocoded1.shp")














