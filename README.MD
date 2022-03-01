# Deep learning models for punch classification in boxing and karate  

This repository contains data and code with our punch classification experiments. 
We are using acceleration sensors measurements and video frames for punch class recognition.

`DOI_10.1109ACCESS.2021.3118038` directory contains data and code for reproducing classification metric results for article 
'Recognition punches in karate using acceleration sensors and convolution neural networks'.  

`code` directory contains:  
- `MoveNetExtractKeypoints.ipynb` - keypoints extraction for punch videos. Save it to `data/keypoints`  
- `RNN-LSTM-GRU.ipynb` - vanilla RNN as baseline.  
- `GRU-NormalizeMidPoint.ipynb` - keypoints coordinates normalized to middle point (btw left and right hips).  

`models` directory contains training results keras and tflite models.  

Starting videos available [here](https://drive.google.com/drive/folders/1UwZPZ7sqkmQrqbCP1ypquv2UHWkk0bj-?usp=sharing).  
On each video man with unique id have done 10 weak (5 left and 5 right hand) and 10 strong punches. 
Total 240 punches in dataset v0.1 We start with only punch class prediction, no power estimation.  

Box punches classes:  
0. no punch,
1. jab (jab left),  
2. cross (jab right),
3. left hook, 
4. right hook, 
5. left uppercut, 
6. right uppercut.   
7. jab (jab left) strong, etc.  

## Docker

Build docker
```
docker build -t punch_dl:v2 .
```

Run docker
```
docker run -p 8888:8888 -v "$(pwd)":/tf punch_dl:v2
```

## Other

IF you want to keep google colab 2 spaces indentation in jupyter notebook, please visit:
https://stackoverflow.com/questions/19068730/how-do-i-change-the-autoindent-to-2-space-in-ipython-notebook
 

## TODO
 
TODO:
 - [x] Extract keypoints frame by frame to *.npy files.
 - [x] Label frames to defined classes.
 - [x] Experiment with recurrent models to punch classification.
 - [x] Add some advanced features, e.g. angles, distances e.t.c.
 - [ ] Experiment with convolution models to punch classification.
 - [ ] Get more videos.
 
 Links:  

Awesome Action Recognition  
https://github.com/jinwchoi/awesome-action-recognition  
