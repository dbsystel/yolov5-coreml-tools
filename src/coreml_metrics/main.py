# Copyright (C) 2021 DB Systel GmbH.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from PIL import Image
import coremltools as ct
import os
import numpy as np
import cv2
import json
from json import JSONEncoder
from pathlib import Path

from objectDetectionMetrics.BoundingBox import BoundingBox
from objectDetectionMetrics.BoundingBoxes import BoundingBoxes
from objectDetectionMetrics.Evaluator import *
from objectDetectionMetrics.utils import *

from argparse import ArgumentParser

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# Model config
IOU_THRESHOLD = 0.6
IOU_THRESHOLD_MODEL = 0.3
CONFIDENCE_THRESHOLD = 0.6

IMAGE_SIZE = 600

LABELS = { "0": "Label1", "1": "Label2", "2": "Label3", "3": "Label4", "4": "Label5"}
IMAGE_FILE_SUFFIXES = (".jpeg", ".jpg")

def main():
    parser = ArgumentParser()
    parser.add_argument('--model-input-path', type=str, dest="model_input_path", default='output/models/yolov5-iOS.mlmodel', help='path to coreml model')
    parser.add_argument('--image-folder', type=str, dest="image_folder", default='data/images', help='path to image root folder')
    parser.add_argument('--label-folder', type=str, dest="label_folder", default='data/labels', help='path to label root folder (folder needs to mirror directory structure of the image folder)')
    parser.add_argument('--metrics_output-directory', type=str, dest="metrics_output_directory", default='output/metrics', help='path to metrics output folder (will be created if it does not exist)')
    opt = parser.parse_args()

    Path(opt.metrics_output_directory).mkdir(parents=True, exist_ok=True)

    allBoundingBoxes = queryFolders(opt.model_input_path, opt.image_folder, opt.label_folder, opt.metrics_output_directory)

# Will scan all subdirectories recursively and write an evulation for all images found in a directory and its child directory
def queryFolders(model, imageFolder, labelsFolder, outputDirectory): 
    allBoundingBoxes = BoundingBoxes()

    for subImageFolder in os.scandir(imageFolder):
        # Recursive call for subfolders
        if subImageFolder.is_dir():
            subBoundingBoxes = queryFolders(model, subImageFolder.path, f"{labelsFolder}/{subImageFolder.name}", outputDirectory)
            allBoundingBoxes.addBoundingBoxes(subBoundingBoxes)

    boundingBoxes = analyseCurrentDir(model, imageFolder, labelsFolder, outputDirectory)
    if boundingBoxes:
        allBoundingBoxes.addBoundingBoxes(boundingBoxes)

    # Check if directory and subdirectories contain any image at all
    if not allBoundingBoxes:
        return

    metricsOutputFolder = imageFolder.replace("data", outputDirectory) + "/metrics"
    Path(metricsOutputFolder).mkdir(parents=True, exist_ok=True)

    print(f"Evaluate {imageFolder}")
    evaluate(allBoundingBoxes, metricsOutputFolder) 
    return allBoundingBoxes 

def analyseCurrentDir(model, imageFolder, labelsFolder, outputDirectory): 
    imageEntries = [imageEntry for imageEntry in os.scandir(imageFolder) 
                               if imageEntry.name.endswith(IMAGE_FILE_SUFFIXES) ]

    if not imageEntries: 
        return

    if not Path(labelsFolder): 
        print(f"Labels folder {labelsFolder} for {imageFolder} doesn't exist")
        return

    detectionOutputFolder = imageFolder.replace("data", outputDirectory) + "/detections"
    Path(detectionOutputFolder).mkdir(parents=True, exist_ok=True)

    imageOutputFolder = imageFolder.replace("data", outputDirectory) + "/images"
    Path(imageOutputFolder).mkdir(parents=True, exist_ok=True)

    detectCoreML(model, imageEntries, detectionOutputFolder)
    boundingBoxes = getBoundingBoxes(labelsFolder, detectionOutputFolder)
    drawBoundingBox(imageEntries, boundingBoxes, imageOutputFolder)

    return boundingBoxes

def evaluate(boundingBoxes, metricsOutputFolder):
    metricsList = Evaluator().PlotPrecisionRecallCurve(
        boundingBoxes,  
        IOUThreshold = IOU_THRESHOLD,
        method = MethodAveragePrecision.EveryPointInterpolation, 
        showAP = True,
        showInterpolatedPrecision = True,
        savePath = metricsOutputFolder,
        showGraphic = False
    )

    with open(f"{metricsOutputFolder}/metrics.json", "w") as metricsFile:
        json.dump(metricsList, metricsFile, cls = NumpyArrayEncoder)
            
def detectCoreML(modelPath, imageEntries, outputFolder):
    model = ct.models.MLModel(modelPath, useCPUOnly=True) 

    for imageEntry in imageEntries:
        inputImage = Image.open(imageEntry.path).resize((640, 640))
        
        out_dict = model.predict({"image": inputImage, "iouThreshold": IOU_THRESHOLD_MODEL, "confidenceThreshold": CONFIDENCE_THRESHOLD}) 

        outFileName = Path(imageEntry.path).stem
        outFilePath = f'{outputFolder}/{outFileName}.txt'

        with open(outFilePath, "w") as outFile:
            for coordinates, confidence in zip(out_dict["coordinates"], out_dict["confidence"]):
                labelMax = confidence.argmax()
                outFile.write("{:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(labelMax, coordinates[0], coordinates[1], coordinates[2], coordinates[3], confidence[labelMax]))

        print(f'Image {outFileName} predicted!') 

# Convert validation and detection YOLO files into Bounding Box Objects
def getBoundingBoxes(valFolder, detectFolder): 
    boundingBoxes = BoundingBoxes()
    addBoundingBoxes(boundingBoxes, valFolder, isGroundTruth = True)
    addBoundingBoxes(boundingBoxes, detectFolder, isGroundTruth = False)
    return boundingBoxes

# Convert label YOLO files into Bounding Box Objects
def addBoundingBoxes(boundingBoxes, labelFolder, isGroundTruth): 
    for labelFileEntry in os.scandir(labelFolder):
        if not labelFileEntry.name.endswith(".txt"):
            continue 

        if not Path(labelFileEntry.path): 
            print(f"Missing label file {labelFileEntry.path}")
            continue
        
        imageName = Path(labelFileEntry.path).stem
        with open(labelFileEntry.path, "r") as labelFile:
            for labelLine in labelFile: 
                labelNumbers = labelLine.split()

                # ignore empty lines 
                if len(labelNumbers) == 0: 
                    continue
                
                if len(labelNumbers) < 5: 
                    print(f'Warning: Not enough values in some line in {groundTruthFolder}/{groundTruthFileName}')
                    continue

                if isGroundTruth:
                    bb = BoundingBox(imageName, LABELS[labelNumbers[0]], float(labelNumbers[1]), float(labelNumbers[2]), float(labelNumbers[3]), float(labelNumbers[4]), CoordinatesType.Relative, (IMAGE_SIZE , IMAGE_SIZE), BBType.GroundTruth, format=BBFormat.XYWH)
                else: 
                    bb = BoundingBox(imageName, LABELS[labelNumbers[0]], float(labelNumbers[1]), float(labelNumbers[2]), float(labelNumbers[3]), float(labelNumbers[4]), CoordinatesType.Relative, (IMAGE_SIZE , IMAGE_SIZE), BBType.Detected, float(labelNumbers[5]), format=BBFormat.XYWH)
                boundingBoxes.addBoundingBox(bb) 

def drawBoundingBox(imageEntries, boundingBoxes, outputFolder):
    for imageEntry in imageEntries:
        imageName = Path(imageEntry.path).stem

        # Read image and resize to model image size
        image = cv2.imread(imageEntry.path)
        (originalHeight, originalWidth) = image.shape[:2]
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        image = boundingBoxes.drawAllBoundingBoxes(image, imageName)

        # Resize image back to original site and write to file
        image = cv2.resize(image, (originalWidth, originalHeight))
        cv2.imwrite(f"{outputFolder}/{imageEntry.name}", image)

        print(f'Image {imageName} boundingBoxes created successfully!') 