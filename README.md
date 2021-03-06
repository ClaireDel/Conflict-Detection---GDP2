# Conflict-Detection---GDP2

Airports are getting smarter – not just on the passenger side, but also on the maintenance and operations side. This is because operations are getting more complex, there are new threats from both humans and autonomous machines, and human operators are often working alongside robots in potentially safety-critical environments.  

The project designs, implements and tests a system for the detection of multiple-humans (crowds) for a surveillance system in a smart airport scenario. 

<p align="center">
  <img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/intro.jpg" width=400 height=auto> <br/> 
  Fig: Model Inference
</p>


## 1. Dataset
[COCO download](http://cocodataset.org/#download): COCO dataset used to train the segmentation model (HRNet)

[Dataset download](https://drive.google.com/drive/folders/1Ezkcq8TW7NJFyS-yKSb8dmNqIFabs2nr):  Original and cleaned dataset (images cropped + black backgrounds)

| ```Repartition``` | ```Original image``` | ```Image cropped``` |
|:---:|:---:|:---:|
|<img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/dataset_repartition.png" width="100%" height="30%">|<img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/image.png" width="70%" height="30%">|<img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/image_crop.png" width="100%" height="30%">|


## 2. Segmentation Phase
This part consists in the first part of the model using HRNet used to get the predicted key points on each human body presents on an image, a video or a video launch. 

### Utilisation <br/> 
[Models required](https://drive.google.com/drive/folders/1Ezkcq8TW7NJFyS-yKSb8dmNqIFabs2nr): Need to download the models folder and place it in the running folder according to the following architecture: 

```
${POSE_ROOT}
├── data
├── experiments
├── lib
├── log
├── models
├── output
├── tools 
├── config
├── core
├── dataset 
├── demo
├── nms
├── utils
├── visualization
├── **model.h5**
├── **model.pickle**
├── **demo.py**
```

### Running <br/> 
The same python code 'demo.py' is used for this part. The model can be updated but the one used by default is in 'demo/inference-config.yaml'. 

The parse can be updated according to the type of results needed:

```
python Segmentation_phase/demo/demo.py
```
- use --webcam when the input is a real-time camera.
- use --video [video-path] when the input is a video.
- use --image [image-path] when the input is an image.
- use --write to save the image, camera or video result.
- use --showFps to show the fps (this fps includes the detection part).

<p align="center">
  <img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/seg_airport.jpg" width=400 height=auto> <br/> 
  Fig: Example of segmentation
</p>




## 3. Classification Phase
### Feature Approach

#### Data Pre-processing <br/> 
- Plotting the key points from pre-cropped images on a blackbackground for the training 
```
python Classification_phase/Features_Approach/preprocessing_vgg.py
```

<p align="center">
  <img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/crop_black.jpg" width=150 height=auto> <br/> 
  Fig: Dataset sample
</p>

#### Training <br/> 
- Training the pre-trained model VGG16 by transfer learning using the pre-processed data
```
python Classification_phase/Features_Approach/training_vgg.py
```
[Model download](https://drive.google.com/drive/folders/1Ezkcq8TW7NJFyS-yKSb8dmNqIFabs2nr): Model trained for using in the testing

#### Testing on pictures <br/> 
- Testing the Neural Network to obtain predictions on images
```
python Classification_phase/Features_Approach/prediction_vgg.py
```

#### Video launch <br/> 
- Testing the Neural Network to obtain real-time predictions on video launch
```
python Classification_phase/Features_Approach/demo_feature_approach.py
```

| ```Pushing``` | ```Demo``` |
|:---:|:---:|
|<img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/pushing_vgg.png" width="90%" height="30%">|<img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/demo_vgg.gif" width="100%" height="30%">|

 
### Keypoint Approach
#### Data Pre-processing <br/> 
- Storing the key points in CSV files after normalization on pre-cropped images
```
python Classification_phase/Keypoints_approach/pre_processing.py
```

| ```Dataset sample``` | ```Data associated in the CSV file``` |
|:---:|:---:|
|<img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/crop.jpg" width="150%" height="30%">|<img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/csv.png" width="150%" height="30%">|


#### Training <br/> 
- Training the classifier (SVM or Multi-Layer Perceptron) using the pre-processed data
```
python Classification_phase/Keypoints_approach/training.py
```

<p align="center">
  <img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/confusionmatrix_svm.png" width=300 height=auto> <br/> 
  Fig: Confusion Matrix for SVM
</p>

- Results
###    
|                | Precision | Recall | F1-score |
|--------------------|------|----------|-------|
| Normal     | 94%|     98% |  96% |  
| Fight    | 95% |     85% |  90% |  
| Accuracy    |     |          |  94% | 
| Macro-average | 94% |     92% |  93% |  
| Weighted-Average | 94% |     94% |  94% |  



- Training the Neural Network Classifier using the pre-processed data
```
python Classification_phase/Keypoints_approach/training_nn.ipynb
```

<p align="center">
  <img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/confusionmatrix_nn.png" width=300 height=auto> <br/> 
  Fig: Confusion Matrix for Neural Network
</p>

- Results
###    
|                | Precision | Recall | F1-score |
|--------------------|------|----------|-------|
| Normal     | 97%|     94% |  95% |  
| Fight    | 91% |     95% |  93% |  
| Accuracy    |     |          |  94% | 
| Macro-average | 94% |     94% |  94% |  
| Weighted-Average | 94% |     94% |  94% |
  


#### Testing on pictures <br/> 
- Testing the SVM classifier to obtain predictions on images
```
python Classification_phase/Keypoints_approach/prediction.py
```
<p align="center">
  <img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/prediction.jpg" width=200 height=auto> 
  <br/> 
  Fig: Image prediction example (Keypoints Approach - SVM)
</p>

- Testing the Neural Network classifier to obtain predictions on images

```
python Classification_phase/Keypoints_approach/testing_nn.py
```
<p align="center">
  <img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/prediction_nn.png" width=300 height=auto> 
  <br/> 
  Fig: Image prediction example (Keypoints Approach - Neural Network)
</p>


#### Video launch <br/> 
- Testing the SVM classifier to obtain real-time predictions on video launch
```
python Classification_phase/Keypoints_approach/demo_keypoint_approach.py
```

| ```Shooting``` | ```Punching``` |
|:---:|:---:|
|<img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/shooting.gif" width="150%" height="30%">|<img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/punching.gif" width="150%" height="30%">|

<p align="center">
  <img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/fight.gif" width="150%" height="30%">
  <br/> 
  Fig: Multi-fight inference (SVM)
<p>
  
- Testing the Neural Network classifier to obtain real-time predictions on video launch
```
Trained model NN: Classification_phase/Keypoints_approach/NN
python Classification_phase/Keypoints_approach/testing_nn.py
```

<p align="center">
  <img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/demo_nn.gif" width="150%" height="30%">
  <br/> 
  Fig: Multi-fight inference (NN)
<p>

### Airport scenarios <br/> 
| ```Scenario 1``` | ```Scenario 2``` |```Scenario 3``` | 
|:---:|:---:|:---:|
|<img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/air1.jpg" width="100%" height="30%">|<img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/air2.jpg" width="80%" height="30%">|<img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/air3.jpg" width="100%" height="30%">|

|```Scenario 4``` |```Scenario 5``` | ```Scenario 6``` |
|:---:|:---:|:---:|
|<img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/air4.jpg" width="110%" height="30%">|<img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/air5.jpg" width="100%" height="30%">|<img src="https://github.com/ClaireDel/Conflict-Detection---GDP2/blob/main/pictures/air6.jpg" width="80%" height="30%">|
