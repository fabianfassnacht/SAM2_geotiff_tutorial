*Tutorial: How to apply SAM2 to a georeferenced geotiff-file and output the masks as vector dataset*

**Introduction**

Hello and welcome to this tutorial which will focus on providing a work-flow to apply meta's SAM2 model to remote sensing data in the **tif-format** and store the results of SAM2 as gepreferenced vector dataset (Shapefile or geopackage).

It will cover all steps from setting-up a Python environment, to installing SAM2 and then apply SAM2 to a very high resolution satellite image and store the results as a vector file.

The python-file required to run the Tutorial can be found here:


The image used in the Tutorial can be downloaded here:



The code to apply SAM2 losely bases on the official tutorial of meta for applying SAM2:

[SAM2 tutorial meta](https://github.com/facebookresearch/sam2/blob/main/notebooks/automatic_mask_generator_example.ipynb)

and was complemented with the help of ChatGPT to make it work with geotiff-files.

**Part 1: Setting-up the Python environment and Spyder**

In Python it is common to set-up environments within which the actual coding and development is accomplished. The idea of an environment is that you install the packages and drivers that you need for your work in sort of an " independent copy" of the original Python installation. The advantage of doing this is that you can have various Python versions and combinations of packages and drivers at the same time. This allows you to ensure that a running work-flow is not corrupted by installing a new package or another Python version you need for another work-flow.

This tutorial works with the Anaconda/Miniconda distribution of Python and the set-up of the environment will be described accordingly. As editor we will use Spyder which is delivered with Anaconda/Miniconda. You can download Miniconda here:

[Anaconda download page](https://docs.anaconda.com/free/miniconda/miniconda-install/)

As first step we will create the environment using the Anaconda prompt. You can open the Anaconda prompt by typing "Anaconda" into the windows search bar (Figure 1).

![Figure 1](https://github.com/fabianfassnacht/PyTorch_Unet_Geotiff_Tutorial/blob/main/Figures_Readme/Fig_01.png)

**Figure 1**


in the now opening command line window you will have to execute several commands. **In some cases, it will be necessary to confirm by pressing the "y" button and enter**. You will find the commands that you have to execute below. Only enter the lines of code **without** the leading # - these lines provide some information to better understand the code. 

conda create --prefix E:/Python_environments/sam python=3.9

conda activate E:\Python_environments\sam

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

pip install git+https://github.com/facebookresearch/segment-anything.git

conda install spyder-kernels=2.5

**Part 2: Download SAM2 checkpoint**


**Part 3: Load Geotiff-file and apply SAM**

blabla
blabla
blabla


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


blabla
blabla
blabla
	
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

blabla
blabla
blabla

	###############################################
	# load geotiff-file and plot it
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
	# load SAM2
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

blabla
blabla
blabla

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

blabla
blabla
blabla

	plt.figure(figsize=(20,20))
	plt.imshow(image)
	show_anns(masks2)
	plt.axis('off')
	plt.show() 

blabla
blabla
blabla

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
