## **Tutorial: How to apply SAM2 to a georeferenced geotiff-file and output the masks as vector dataset**

**Introduction**

Hello and welcome to this tutorial which will focus on providing a work-flow to apply meta's SAM2 model to remote sensing data in the **tif-format** and store the results of SAM2 as georeferenced vector dataset (Shapefile or geopackage).

It will cover all steps from setting-up a Python environment, to installing SAM2 and then apply SAM2 to a very high resolution satellite image and store the results as a vector file.

The python-file required to run the Tutorial can be found here:

[Python file](https://github.com/fabianfassnacht/SAM2_geotiff_tutorial/tree/main/Python_code)

The image used in the Tutorial can be downloaded here:

[Satellite image](https://drive.google.com/file/d/1n5ilJE9S2d4xz8PyGXb92syWXu5DuJ19/view?usp=sharing)

Be aware that depending on the power of your computer and the avaialability of a GPU, this image (even though it is quite small) might need a long time on your computer to be processed by SAM. You can also use any other geotiff-file you have available and in case you have a slow computer, it might be best to work with a really small subset.

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

	# create the new Python environment (at least Python version 3.9.21) - you will have to adapt the path according
 	# to your computer
	conda create --prefix E:/Python_environments/sam python=3.9.21

 	# activate the just created Python environment
	conda activate E:/Python_environments/sam

   	# it may be necessary to also install the cv2 (open cv) package and several other 
    	# auxiliary Python packages:
    	conda install -c conda-forge opencv
     	conda install shapely
	conda install geopandas
 	conda install rasterio
	conda install matplotlib

 	# other key packages include pytorch, torchvision, torchaudio, as well as cuda drivers
	conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

 	# install spyder to work with the spyder editor (feel free to use another one
  	# if you prefer another one - then you do not have to take this step)
	conda install spyder


     
 

**Part 2: Download SAM2 checkpoint**

As next step you have to download a trained version of SAM2 which in form of a so-called "checkpoint" which you can find here:

[SAM 2 checkpoint download](https://github.com/facebookresearch/sam2?tab=readme-ov-file#model-description)

The checkpoint used in this tutorial (downloaded from meta's page as well) can also be found here:

[SAM 2 checkpoint used in this tutorial](https://drive.google.com/file/d/1R8_eZ2yI6SJHyT7P8xqZ1QZwL-La8wcK/view?usp=sharing)


**Part 3: Load Geotiff-file and apply SAM**

As next step, we run the code to load a Geotiff-file and apply SAM. For this we first of all have to make sure that all packages are installed. We can run the code below by marking the corresponding text in spyder and then press the button marked in Figure 2.

![Figure 2](https://github.com/fabianfassnacht/SAM2_geotiff_tutorial/blob/main/Images/SAM_02.png)

**Figure 2**

If this results in some sort of error message we will have to install the missing packages by running the lines of code that are "outcommented" with the hashtag sign below. Normally, this should not be the case since we installed all packages already in the Anaconda prompt above - except for the SAM packages which we will need to install for sure. Be aware that there are two option to install packages in Python - one is using the command "conda install packagename" (works in Anaconda prompt only as far as I know) and one is "!pip install packagename". Conda should always be prepared when working in a virtual Anaconda enviroment as we do at the moment since the package will then only be installed in the current environment. Running !pip may install the package globally, affecting the entire Conda setup and this can cause troubles not only for your current project and environment. At the same time, in some cases there will be no option since some packages may not be available with the conda command. One other trick you can apply is that you can sometimes use the additional "conda-forge" attribute which then also searches for packages outside of the "officially approved" conda packages. Some community built packages and most recent versions of packages can only be found using this argument.


Next we will have to install segment-anything from meta using:

	pip install git+https://github.com/facebookresearch/segment-anything.git

Then we call all required packages:

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

If you forgot to install some packages you could run for example this code to install the shapely package - but see warning above:

	#!pip install shapely

Next we define a function that allows to plot the input image along with the masks created by SAM.
	
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

Now we load the tif-file and plot it.

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

This should result in the following plot if you change to the "Plots window in Spyder":

![Figure 3](https://github.com/fabianfassnacht/SAM2_geotiff_tutorial/blob/main/Images/SAM_03.png)

**Figure 3**

To be able to process the tif-file with SAM, we need to make sure that it is stored in the right data type format. So in this case we have to transform the image from a unsigned integer with 16 bit (uint16) to one with 8 bit (uint8)

	# Normalize raster to uint8 if needed
	if rgb_image.dtype == np.uint16:
	    rgb_image = cv2.normalize(rgb_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

Now we load the SAM package and the checkpoint (that is a trained version of the SAM algorithm) - for this we have to define the path on our computer to where we have saved the checkpoint you downloaded from the link provided above.

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


Now we are almost ready to apply the SAM model to our image. We now have the option to change some settings of the SAM algorithm. I asked ChatGPT for some explanations of the settings and these are as shown below. Be aware that we do not use all of the parameters in the code below and that some settings have different values then written in the explanations below:

### **Parameters**

1. **`points_per_side=64`**
   - Determines the grid density for sampling points over the input image. 
   - **Explanation**: A grid with 64 points per side means \( 64 \times 64 = 4096 \) sampling points across the image.
   - **Effect**: A higher value increases mask accuracy (by sampling more points) but also raises computation costs.

---

2. **`points_per_batch=128`**
   - Specifies the number of sampled points processed in a single batch.
   - **Explanation**: If there are 4096 points to process (from `points_per_side=64`), the model processes these points in batches of 128.
   - **Effect**: Controls memory usage and speed; smaller batches are memory-efficient but slower.

---

3. **`pred_iou_thresh=0.7`**
   - The Intersection over Union (IoU) threshold for selecting predicted masks.
   - **Explanation**: Masks with predicted IoU scores below this threshold are discarded.
   - **Effect**: A higher threshold results in fewer but higher-quality masks.

---

4. **`stability_score_thresh=0.92`**
   - The threshold for the mask's stability score.
   - **Explanation**: Stability scores measure how consistent a mask is when slightly perturbed. Only masks with scores above this threshold are retained.
   - **Effect**: Higher values favor more stable masks, potentially discarding less reliable regions.

---

5. **`stability_score_offset=0.7`**
   - Offset applied when calculating stability scores.
   - **Explanation**: This value tweaks how mask stability is measured. It shifts the evaluation window for determining stability.
   - **Effect**: Adjusts sensitivity to mask consistency.

---

6. **`crop_n_layers=1`**
   - Number of times the image is cropped to generate masks.
   - **Explanation**: The image is cropped into smaller regions to refine mask generation. `1` means one layer of cropping.
   - **Effect**: Higher values increase mask refinement but also computation time.

---

7. **`box_nms_thresh=0.7`**
   - Non-Maximum Suppression (NMS) threshold for overlapping bounding boxes of masks.
   - **Explanation**: If two masks have overlapping boxes with an IoU above this threshold, one of them is discarded.
   - **Effect**: Reduces redundant masks, ensuring only unique regions are segmented.

---

8. **`crop_n_points_downscale_factor=2`**
   - Downscaling factor for the number of points used during cropping.
   - **Explanation**: When generating masks for cropped regions, the number of points is reduced by this factor.
   - **Effect**: Balances mask quality in cropped regions with computational efficiency.

---

9. **`min_mask_region_area=25.0`**
    - Minimum area (in pixels) for a mask to be considered valid.
    - **Explanation**: Masks smaller than this area are discarded as they might represent noise or irrelevant details.
    - **Effect**: Prevents generating overly small and insignificant masks.

---

10. **`use_m2m=True`**
    - Whether to use the "mask-to-mask" (M2M) refinement process.
    - **Explanation**: M2M refines generated masks by comparing and merging overlapping ones.
    - **Effect**: Enhances mask quality by reducing redundancy and filling gaps.

---

### **Summary**
These parameters configure the **mask generation process** in SAM2, balancing **quality**, **stability**, and **computational efficiency**. Adjusting these allows tailoring the segmentation process to the needs of your application, whether prioritizing speed, accuracy, or memory usage.

---

	###############################################
	# adjust settings of SAM and apply it to image
	###############################################

	mask_generator_2 = SamAutomaticMaskGenerator(
	    model=sam,
	    points_per_side=64,
	    pred_iou_thresh=0.86,
	    stability_score_thresh=0.92,
	    crop_n_layers=1,
	    crop_n_points_downscale_factor=2,
	    min_mask_region_area=20,  # Requires open-cv to run post-processing
	)



Then we apply the model with the following code - this process can now take up to several minutes. If you check the task manager of your computer, you will see that either your CPU or GPU will be very busy.

	masks2 = mask_generator_2.generate(rgb_image)

Once this code has run successfully, you can plot the results running this code:

	plt.figure(figsize=(20,20))
	plt.imshow(image)
	show_anns(masks2)
	plt.axis('off')
	plt.show() 

This should lead to a plot that looks similar to this (depending on the settings you applied):

![Figure 4](https://github.com/fabianfassnacht/SAM2_geotiff_tutorial/blob/main/Images/SAM_04.png)

**Figure 4**


As last step, we can now save the created mask as geocoded vector-files to the harddisc. For this we use the following code:


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


You can either save as Shapefile or geopackage or in both data formats:

	# Create GeoDataFrame with CRS from the raster
	gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)

	# Save to GeoPackage or Shapefile
	output_path = 'E:/8_SAM_tutorial/'
	gdf.to_file(f"{output_path}/output_geocoded1.gpkg", driver="GPKG")
	gdf.to_file(f"{output_path}/output_geocoded1.shp")


We can now have a look at the data in QGIS or another GIS environment. In the plot below, I changed the visualization settings by giving each polygon an own color and setting the opacity to 35%.

![Figure 5](https://github.com/fabianfassnacht/SAM2_geotiff_tutorial/blob/main/Images/SAM_05.png)

**Figure 5**
