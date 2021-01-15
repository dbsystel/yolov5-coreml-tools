# yolov5 - CoreML Tools

The scripts in this repo can be used to export a YOLOv5 model to coreml and benchmark it. 

## Dependencies

* python 3.8
* poetry

Other dependencies are installed by poetry automatically with:
```console 
$ poetry install
```

## Export to CoreML 

We use the unified conversion api of [coremltools 4.0](https://coremltools.readme.io/docs) to convert a YOLOv5 model to CoreML. Additionally we add an export layer, so the model integrates well with [Apple's Vision Framework](https://developer.apple.com/documentation/vision). 

### Limitations

For optimal integration in Swift you need at least `iOS 13` as the export layer with NMS is otherwise not supported. It is possible though to use `iOS 12` if you implement NMS manually in Swift.

The models always need the original source code, unless you do have a torchscript model and therefore skip the tracing step in the script. Currently we use our own fork of YOLOv5 with some modifications regarding Neural Engine Optimizations. The fork is using YOLOv5 Version 2.0, so you will need a model trained with Version 2.0. 

Experience has shown that one needs to be very careful with the versions of libaries used. In particular the PyTorch version. It's recommended to not change the version of the libaries used. If you need for some reasons a newer PyTorch version, check the [github page of coremltools](https://github.com/apple/coremltools/issues) for open issues regarding the PyTorch version you want to use, in the past most recents versions where not compatible.

### Usage 

First, some values in the script should be changed according to your needs. In `src/coreml_export/main.py` you'll find some global variabled, which are specific to the concrete model you use: 

* `classLabels` -> This is the list of labels your model recognized, all pretrained models of YOLOv5 used the [coco dataset](https://cocodataset.org/#home) and therefore regonize 80 Labels
* `anchors` -> These depend on the model version you use (s, m, l, x) and you will find these in the according `yolo<Version>.yml` file in the `yolov5` repository (Use the files from the correct yolov5 version!). 
* `reverseModel` -> Some models have their strides and anchors in reversed order. This variable exists for convinience to quickly change the order of those.

To run the script use the command: 
```console 
$ poetry run coreml-export
```
Run it with `-h` to get a list of optional arguments to customize model input & output paths / names and other things. 

## Helper Export Scripts

### Testing

There is a simple script to test the exported model with some sample images in `src/coreml_export/test.py`. You should check if the predictions are similar to those of the original PyTorch model. Be aware that the predictions will be slightly different though. If there are huge differences this might be a hint that you need to set `reverseModel` accordingly. 

To run the script use the command: 
```console 
$ poetry run coreml-test
```

### Debugging / Fixing Issues 

Most important is that the model runs fully on the Neural Engine. There is no official documentation, but take a look at [Everything we actually know about the Apple Neural Engine (ANE)](https://github.com/hollance/neural-engine). 
In  `src/coreml_export/snippets.py` you might find a few helpful snippets to temporarily (!) change layers, parameters or other things of the model and test how it influences performance.  

## CoreML Metrics 

This makes heavy use of the library [Object Detection Metrics Library](https://github.com/rafaelpadilla/Object-Detection-Metrics) developed by @rafaelpadilla. 

The Metrics script in  `src/coreml_metrics/main.py` can be used to benchmark a CoreML Model. 

