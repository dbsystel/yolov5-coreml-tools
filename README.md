# yolov5 - CoreML Tools

The scripts in this repo can be used to export a YOLOv5 model to coreml and benchmark it. 

## Dependencies

* python 3.8.x
* poetry

Other dependencies are installed by poetry automatically with:
```console 
$ poetry install
```
**Note**: It assumes that you've cloned the yolov5 repo to `../yolov5` relative to the project and that you have added a `setup.py` file, so poetry installs the dependencies of YOLOv5. See [Issue #2525](https://github.com/ultralytics/yolov5/issues/2525#issuecomment-821525523).

It's recommended to use the version as specified. CoreML Tools only works with certain PyTorch Versions, which are only available for certain Python versions. You might want to consider using pyenv:
```
$ pyenv install 3.8.6
$ pyenv global 3.8.6
$ poetry install
```

## Export to CoreML 

We use the unified conversion api of [coremltools 4.0](https://coremltools.readme.io/docs) to convert a YOLOv5 model to CoreML. Additionally we add an export layer, so the model integrates well with [Apple's Vision Framework](https://developer.apple.com/documentation/vision). 

### Limitations

For optimal integration in Swift you need at least `iOS 13` as the export layer with NMS is otherwise not supported. It is possible though to use `iOS 12` if you implement NMS manually in Swift.

Experience has shown that one needs to be very careful with the versions of libaries used. In particular the PyTorch version. It's recommended to not change the version of the libaries used. If you need for some reasons a newer PyTorch version, check the [github page of coremltools](https://github.com/apple/coremltools/issues) for open issues regarding the PyTorch version you want to use, in the past most recents versions where not compatible.

#### YOLOv5 Version 
The models always need the original source code, unless you do have a torchscript model and therefore skip the tracing step in the script. At the time writing, we use YOLOv5 version 2.0 and have only tested it with this version.

**Note**: It has a huge impact on performance if the model runs on the NeuralEngine or the CPU / GPU (or switches between them) on your device. Unfortunately, there is no documentation which model layers can run on the neural engine and which not ([some infos here](https://github.com/hollance/neural-engine)). With yolov5 version 2, 3 and 4 there were problems with the SPP Layers with kernel sizes bigger than 7, so we replaced them and retrained the model. On a recent device YOLOv5s should be around 20ms / detection. 
See [Issue 2526](https://github.com/ultralytics/yolov5/issues/2526#issuecomment-823059344)

With yolov5 version2 we found out that SPP Layers with kernel sizes bigger 7 are not supported, so you might want to change the model configration, so it uses smaller kernel sizes before you train it. The smallest YOLOv5s should be around 20ms / detection if optimized. 
Please open an issue if you have problems with other layers (for instance in newer YOLOv5 versions)!


### Usage 

First, some values in the script should be changed according to your needs. In `src/coreml_export/main.py` you'll find some global variables, which are specific to the concrete model you use: 

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

This makes heavy use of the library [Object Detection Metrics Library](https://github.com/rafaelpadilla/Object-Detection-Metrics) developed by @rafaelpadilla under the MIT License. 
The library is included in the `objectionDetectionMetrics` subfolder with some small adjustments.

The Metrics script in  `src/coreml_metrics/main.py` can be used to benchmark a CoreML Model. It would calculate a precision x recall curve for every label and every folder of images. 
See [here for a detailed explanation](https://github.com/rafaelpadilla/Object-Detection-Metrics#important-definitions).

### Usage 

First, you need some images with ground truth data to benchmark the model. The images can be in a nested folder structure to allow benchmarking categories of images. It's just important that you exactly mirror the folder structure with your ground truth data. Example structure: 
```
- data 
    - images
        - sharp_images
            - black_white_images
                - image1.jpg 
                - image2.jpg
            - colored_images 
                - image3.jpg 
        - unsharp_images 
            - image4.jpg 
            - image5.jpg 
    - labels 
        - sharp_images
            - black_white_images
                - image1.txt
                - image2.txt
            - colored_images 
                - image3.txt 
        - unsharp_images 
            - image4.txt 
            - image5.txt 
```
The script will then output detections for every image and one graph for each tag for each folder. So there will be one graph for all black_white_images, one for all colored_images, one for all sharp images ... 

Furthermore, some values in the script should be changed according to your needs. In `src/coreml_metrics/main.py` you'll find some global variables, which are specific to the concrete model you use: 

* `classLabels` -> This is the list of labels your model recognized, all pretrained models of YOLOv5 used the [coco dataset](https://cocodataset.org/#home) and therefore regonize 80 Labels

To run the script use the command: 
```console 
$ poetry run coreml-metrics
```
