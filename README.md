# Conflict-Detection---GDP2

... INTRO
Requirements: Tensorflow for vgg, ...
Reupload

## 1. Dataset
... drive final dataset
... drive COCO
[COCO download](http://cocodataset.org/#download), for COCO keypoints training and validation
...cropped pictures
...cropped + black-background

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

<p align="center">
  <img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/seg_airport.jpg" width=400 height=auto>
  Fig: Example of segmentation
</p>




## 3. Classification Phase
### Feature Approach

#### Data Pre-processing <br/> 
- Plotting the key points from pre-cropped images on a blackbackground for the training 
```
python Classification_phase/Feature_Approach/preprocessing_vgg.py
```

#### Training <br/> 
- Training the pre-trained model VGG16 by transfer learning using the pre-processed data
```
python Classification_phase/Feature_Approach/training_vgg.py
```
...pickles


#### Testing on pictures <br/> 
- Testing the Neural Network to obtain predictions on images
```
python Classification_phase/Feature_Approach/prediction_vgg.py
```

#### Video launch <br/> 
- Testing the Neural Network to obtain real-time predictions on video launch
```
python Classification_phase/Feature_Approach/demo_feature_approach.py
```

...NOTES:  Demo photos and videos gif
![1 person](inference_1.jpg)
Fig: 1 person inference

 <br/> 
 
### Keypoint Approach
#### Data Pre-processing <br/> 
- Storing the key points in CSV files after normalization on pre-cropped images
```
python Classification_phase/pre_processing.py
```
... csv files

#### Training <br/> 
- Training the classifier (SVM or Multi-Layer Perceptron) using the pre-processed data
```
python Classification_phase/training.py
```

#### Testing on pictures <br/> 
- Testing the classifier (SVM or Multi-Layer Perceptron) to obtain predictions on images
```
python Classification_phase/prediction.py
```

#### Video launch <br/> 
- Testing the classifier (SVM or Multi-Layer Perceptron) to obtain real-time predictions on video launch
```
python Classification_phase/demo_keypoint_approach.py
```

...NOTES: Demo photos and videos gif
![1 person](inference_1.jpg)
Fig: 1 person inference
