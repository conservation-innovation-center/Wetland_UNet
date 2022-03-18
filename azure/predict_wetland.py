# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 10:12:07 2021

@author: MEvans
"""

from azureml.core import Run, Workspace, Model
import os
import glob
from os.path import join
import tensorflow as tf
from tensorflow.python.keras import models
import argparse
import sys
import json

# import custom modules
sys.path.append(os.path.join(sys.path[0], 'scv'))

from scv.utils.model_tools import get_binary_model, weighted_bce
from scv.utils.prediction_tools import makePredDataset, make_array_predictions, write_geotiff_prediction, write_geotiff_predictions

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type = str, required = True, help = 'directory containing test image(s) and mixer')
parser.add_argument('--model_id', type = str, required = True, help = 'model used to make predictions')
parser.add_argument('--aoi', type = str, required = True, help = 'basename of study area being predicted')
args = parser.parse_args()

# Define some global variabes
# get the run context
run = Run.get_context()
exp = run.experiment
ws = exp.workspace

# specify surface layers
lidar = ['lidar_intensity']
geomorphon = ["geomorphons"]

# Specify inputs (Sentinel bands) to the model
opticalBands = ['B3', 'B4', 'B5', 'B6']
thermalBands = ['B8', 'B11', 'B12']
senBands = opticalBands + thermalBands

# get band names for three seasons
seasonalBands = [[band+'_summer', band + '_fall', band + '_spring'] for band in senBands]

# specify NAIP bands
naipBands = ['R', 'G', 'B', 'N']
aoi = args.aoi

if 'wlidar' in args.model_id:
    BANDS = [item for sublist in seasonalBands for item in sublist] + naipBands + lidar
    ONE_HOT = None
    DEPTH = len(BANDS)
    name = 'wlidar'
elif 'wgeomorphon' in args.model_id:
    BANDS = [item for sublist in seasonalBands for item in sublist] + naipBands + geomorphon
    ONE_HOT = {'geomorphons':11}
    DEPTH = len(BANDS)+sum(ONE_HOT.values())-len(ONE_HOT.values())
    name = 'wgeomorphon'
elif 'full' in args.model_id:
    BANDS = [item for sublist in seasonalBands for item in sublist] + naipBands + lidar + geomorphon
    ONE_HOT = {'geomorphons':11}
    DEPTH = len(BANDS)+sum(ONE_HOT.values())-len(ONE_HOT.values())
    name = 'full'
else:
    BANDS = [item for sublist in seasonalBands for item in sublist] + naipBands
    ONE_HOT = None
    DEPTH = len(BANDS)
    name = 'basic'

print('name is ', name)
FEATURES = BANDS# + MORPHS + [RESPONSE]
print(FEATURES)
# Specify the size and shape of patches expected by the model.
KERNEL_SIZE = 256
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]

OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)

METRICS = {
    'logits':[tf.keras.metrics.MeanSquaredError(name='mse'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')],
    'classes':[tf.keras.metrics.MeanIoU(num_classes=2, name = 'mean_iou')]
    }

#testFiles = []
#
#for root, dirs, files in os.walk(dataset_mount_folder):
#    for f in files:
#        testFiles.append(join(root, f))

# get data to run predictions on
predFiles = glob.glob(join(args.data_dir, '*.gz'))
jsonFile = glob.glob(join(args.data_dir, '*.json'))
print(jsonFile[0])   
#predData = makePredDataset(predFiles, BANDS, one_hot = ONE_HOT)

model_dir = Model.get_model_path(args.model_id, _workspace = ws)
weight_path = glob.glob(join(model_dir, '*.hdf5'))
model_path = glob.glob(join(model_dir, '*.h5'))
print(weight_path)

def get_weighted_bce(y_true,y_pred):
  return weighted_bce(y_true, y_pred, 1)

# m = models.load_model('azureml-models/wetland-unet-basic/5/outputs/unet256.h5', custom_objects = {'get_weighted_bce': get_weighted_bce})
# m = get_model(depth = DEPTH, optim = OPTIMIZER, loss = get_weighted_bce, mets = METRICS, bias = None)
m = models.load_model(model_path[0], custom_objects = {'get_weighted_bce':get_weighted_bce})
m.load_weights(weight_path[0])

# create special folders './outputs' and './logs' which automatically get saved
os.makedirs('outputs', exist_ok = True)
os.makedirs('logs', exist_ok = True)
out_dir = './outputs'
log_dir = './logs'

preDataset = makePredDataset(
    file_list = predFiles,
    features = BANDS,
    kernel_shape = [256, 256],
    kernel_buffer = [128, 128],
    axes = [2],
    splits = None,
    moments = None,
    one_hot = ONE_HOT)

write_geotiff_predictions(
    imageDataset = preDataset,
    model =  m,
    jsonFile = jsonFile[0],
    outImgBase = f'{aoi}_{name}',
    outImgPath = out_dir, 
    kernel_buffer = [128,128])
    
#preds = make_array_predictions(imageDataset = predData, model = m, jsonFile = jsonFile[0])
#prob = preds[:, :, 0]
#write_geotiff_prediction(prob, jsonFile[0], f'{out_dir}/DE_test_{name}')