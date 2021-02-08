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

import torch

# Snippet to only include a certain number of layers (debugging)
def includeNumberOfLayers(model, numberOfLayers):
    model.model = torch.nn.Sequential(*[model.model[i] for i in range(numberOfLayers)])

# Snippet to change some attributes of a specific layer (Note: Mostly for debugging as it will often affect model performance)
def fixModel(modelSpec):
    kernelSizes = [3, 5, 7]
    poolingCount = 0 

    for layer in modelSpec.neuralNetwork.layers:
        if layer.WhichOneof("layer") == "pooling":
            layer.pooling.kernelSize[:] = [kernelSizes[poolingCount], kernelSizes[poolingCount]] #[5, 5]

            layer.pooling.valid.paddingAmounts.borderAmounts[0].startEdgeSize = kernelSizes[poolingCount] // 2#2
            layer.pooling.valid.paddingAmounts.borderAmounts[0].endEdgeSize =  kernelSizes[poolingCount] // 2#2
            layer.pooling.valid.paddingAmounts.borderAmounts[1].startEdgeSize = kernelSizes[poolingCount] // 2# 2
            layer.pooling.valid.paddingAmounts.borderAmounts[1].endEdgeSize = kernelSizes[poolingCount] // 2#2

            poolingCount += 1

# Snippet to replace a part from one model with a part from another model
def fixModel2(modelSpecOld, modelSpec):
    del modelSpec.neuralNetwork.layers[0:24] 
    for i in range(8):
        modelSpec.neuralNetwork.layers.insert(i,  modelSpecOld.neuralNetwork.layers[i])

    # modelSpec.neuralNetwork.layers[8].input[:] = ["325", "326", "327", "328"]
    modelSpec.neuralNetwork.layers[8].input[:] = ["319", "320", "321", "322"]

    # for layer in modelSpec.neuralNetwork.layers:       
    #     print(layer.name)


    # ct.models.utils.save_spec(modelSpec, f'performanceTest.mlmodel')
    # return


# Problematic part of yolov5 model for coreml (kernel size > 5 doesn't run on Neural Engine), can be used to test a specific part of a model and find the problematic layer / attribute
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(torch.nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = torch.nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = torch.nn.BatchNorm2d(c2)
        self.act = torch.nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
        #return x # self.conv(x)

    def fuseforward(self, x):
        return self.act(self.conv(x))


class SPP(torch.nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(9,)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k)), c2, 1, 1)
        self.m = torch.nn.ModuleList([torch.nn.MaxPool2d(kernel_size=x, stride=1, padding= x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        #return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
        return self.cv2(torch.cat([m(x) for m in self.m], 1))

def main(): 
    model = SPP(512, 512)
    torch.save(model, "experiments/experiment.pt")