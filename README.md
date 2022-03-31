# Conflict-Detection---GDP2

... INTRO
Requirements: Tensorflow for vgg, ...
REUPLOAD
read me intermediaire
fichiers à la source à chaque fois
expliquer les chemins et param

<p align="center">
  <img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/intro.jpg" width=400 height=auto> <br/> 
  Fig: Model Inference
</p>


## 1. Dataset
[COCO download](http://cocodataset.org/#download): COCO dataset used to train the segmentation model (HRNet)

[Dataset download](https://drive.google.com/drive/folders/1Ezkcq8TW7NJFyS-yKSb8dmNqIFabs2nr):  Original and cleaned dataset (images cropped + black backgrounds)

| ```Original image``` | ```Image cropped``` |
|:---:|:---:|
|<img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/image.png" width="70%" height="30%">|<img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/image_crop.png" width="100%" height="30%">|



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

...NOTES: 
model folder in the drive: ...

<p align="center">
  <img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/seg_airport.jpg" width=400 height=auto> <br/> 
  Fig: Example of segmentation
</p>




## 3. Classification Phase
### Feature Approach

#### Data Pre-processing <br/> 
- Plotting the key points from pre-cropped images on a blackbackground for the training 
```
python Classification_phase/Feature_Approach/preprocessing_vgg.py
```

<p align="center">
  <img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/crop_black.jpg" width=150 height=auto> <br/> 
  Fig: Dataset sample
</p>

#### Training <br/> 
- Training the pre-trained model VGG16 by transfer learning using the pre-processed data
```
python Classification_phase/Feature_Approach/training_vgg.py
```
[Model download](https://drive.google.com/drive/folders/1Ezkcq8TW7NJFyS-yKSb8dmNqIFabs2nr): Model trained for using in the testing

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

| ```Dataset sample``` | ```Data associated in the CSV file``` |
|:---:|:---:|
|<img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/crop.jpg" width="150%" height="30%">|<img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/csv.png" width="150%" height="30%">|


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
<p align="center">
  <img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/prediction.jpg" width=200 height=auto> 
  <br/> 
  Fig: Image prediction example
</p>

#### Video launch <br/> 
- Testing the classifier (SVM or Multi-Layer Perceptron) to obtain real-time predictions on video launch
```
python Classification_phase/demo_keypoint_approach.py
```

| ```Shooting``` | ```Punching``` |
|:---:|:---:|
|<img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/shooting.gif" width="150%" height="30%">|<img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/punching.gif" width="150%" height="30%">|

<p align="center">
  <img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/fight.gif" width="150%" height="30%">
  <br/> 
  Fig: Multi-fight inference
<p>
