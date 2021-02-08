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
from pathlib import Path
import coremltools as ct
import os

IMAGE_FOLDER = "data/images"

def main():
    model = ct.models.MLModel("output/models/yolov5-iOS.mlmodel")
    # model_16 = ct.models.MLModel("output/models/yolov5-iOS.mlmodel")
    # model_8 = ct.models.MLModel("output/models/yolov5-iOS.mlmodel")

    in_dicts = []
    imagePaths = []
    for imagePath in Path(IMAGE_FOLDER).rglob('*.jpg'):
        imagePaths.append(imagePath)
        in_dicts.append({"image": Image.open(imagePath).resize((640, 640))})

    for i, in_dict in enumerate(in_dicts):
        out_dict = model.predict(in_dict) 

        print(imagePaths[i])
        print("Confidences: ")
        print(out_dict["confidence"])
        print("Bounding Box")
        print(out_dict["coordinates"])
        print()


    # Compare models 
    # ct.models.neural_network.quantization_utils.compare_models(model, model_16, in_dicts)
    # ct.models.neural_network.quantization_utils.compare_models(model, model_8, in_dicts)