# Conflict-Detection---GDP2

... INTRO

## 1. Dataset

## 2. Segmentation Phase
This part consists in the first part of the model using HRNet used to get the predicted key points on each human body presents on an image, a video or a video launch. 
The same python code 'demo.py' is used for this. The model can be updated but the one used by default is in demo/inference-config.yaml

The parse can be updated according to the type of results needed:

```
python demo/demo.py 
- use --webcam when the input is a real-time camera.
- use --video [video-path] when the input is a video.
- use --image [image-path] when the input is an image.
- use --write to save the image, camera or video result.
- use --showFps to show the fps (this fps includes the detection part).
```

...NOTES: Demo photos and videos 

## 3. Classification Phase
### Feature Approach
- Training
- Testing
- Evaluation
NOTES: prepro / training / plot photos / video
crop

### Keypoint Approach
- Training
- Testing
- Evaluation
NOTES: prepro / training / plot photos / video
csv
crop