# Conflict-Detection---GDP2

... INTRO

## 1. Dataset
... drive final dataset
... drive COCO
[COCO download](http://cocodataset.org/#download), for COCO keypoints training and validation


## 2. Segmentation Phase
This part consists in the first part of the model using HRNet used to get the predicted key points on each human body presents on an image, a video or a video launch. 
The same python code 'demo.py' is used for this. The model can be updated but the one used by default is in demo/inference-config.yaml

The parse can be updated according to the type of results needed:

```
python Segmentation_phase/demo/demo.py
```
- use --webcam when the input is a real-time camera.
- use --video [video-path] when the input is a video.
- use --image [image-path] when the input is an image.
- use --write to save the image, camera or video result.
- use --showFps to show the fps (this fps includes the detection part).

...NOTES: Demo photos and videos gif
model folder in the drive: ...
![1 person](inference_1.jpg)
Fig: 1 person inference

## 3. Classification Phase
### Feature Approach
- Preprocessing
- Training
- Testing / test set
- Evaluation

...NOTES:  Demo photos and videos gif
crop
pickles
![1 person](inference_1.jpg)
Fig: 1 person inference


### Keypoint Approach
- Data Pre-processing <br/> 
```
python Classification_phase/pre_processing.py
```
Storing the key points in CSV files after normalization on pre-cropped images

- Training
SVM
MLP

- Testing / test set
- Evaluation

...NOTES: Demo photos and videos gif
crop
csv
pickles 
![1 person](inference_1.jpg)
Fig: 1 person inference
