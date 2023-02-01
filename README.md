# Wetland_UNet
This repo was developed by Conservation Innovation Center of Chesapeake Conservancy and Defenders of Wildlife. It contains code used to train and run predictions from UNet models that can delineate wetlands using remote sensing data input including bands from Sentinel-2 LiDAR and geomorphons.

## Data
The data used in this repo are in TFRecords format. The data was exported using Google Earth Engine in Google Colab environment, and are available in an [Open Science Framework repository](https://osf.io/ts5eu/).

## Scripts
The python script used to train UNet model are available in this repo. To train models in an Azure ML Studio environment, clone this repository and navigate to the /azure directory. Then clone the https://github.com/mjevans26/Satellite_ComputerVision.git repo, which contains necessary modules. Follow the steps in the Setup.ipynb to create a suitable MLStudio environment and resources, then use Train.ipynb to train a model.

## Model
The trained Keras model and weights are available in an [Open Science Framework repository](https://osf.io/ts5eu/).

## Storymap
Electric Power Research Institute has developed a storymap out of this work. The storymap provides details of the data, method, model evaluation scores, types of models, etc. The storymap is availabe here: https://storymaps.arcgis.com/stories/4f98297b48a94efbbbe0199681539980

<img width="1727" alt="Screen Shot 2021-12-15 at 1 21 27 AM" src="https://user-images.githubusercontent.com/14167540/146134249-eb17f3af-237d-4222-9497-4579876cb769.png">


## Webapp
Our model output can be compared against wetland reference data. Conservation Innovation Center of Chesapeake Conservancy has developed a webapp where you can load various layers of input data of the model. The webapp is available here: https://cicgis.org/portal/apps/webappviewer/index.html?id=7bd206c0a2f0462ea6a821d9c4c2de68

https://user-images.githubusercontent.com/14167540/146135799-629dcb68-c258-4172-874f-9a351d488f6f.mov
