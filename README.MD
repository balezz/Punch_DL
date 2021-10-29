# Deep learning models for punch classification in boxing and karate  

This repository contains data and code with our punch classification experiments. 
We are using acceleration sensors measurements and video frames for punch class recognition.

`DOI_10.1109ACCESS.2021.3118038` - data and code for reproducing classification metric results for article 
'Recognition punches in karate using acceleration sensors and convolution neural networks.  
 
`MoveNetExtractKeypoints.ipynb` - keypoints extraction for punch videos.  
`RNN-LSTM.ipynb` - vanilla RNN as baseline.  
`LSTM-NormalizeMidPoint.ipynb` - keypoints coordinates normalized to middle point (btw left and right hips).  

Starting videos available [here](https://drive.google.com/drive/folders/1UwZPZ7sqkmQrqbCP1ypquv2UHWkk0bj-?usp=sharing).  
On each video two man (id0 and id1) have done 10 weak (5 left and 5 right hand) and 10 strong punches. 
Total 240 punches in dataset v0.1 We start with only punch class prediction, no power estimation.  

Box punches classes:

0. no punch,
1. jab (jab left),  
2. cross (jab right),
3. left hook, 
4. right hook, 
5. left uppercut, 
6. right uppercut.   


## Docker

Build docker
```
docker build -t punch_dl:v1 .
```

Run docker
```
docker run -p 8888:8888 -v "$(pwd)":/tf punch_dl:v1
```

## Other

IF you want to keep google colab 2 spaces indentation in jupyter notebook, please visit:
https://stackoverflow.com/questions/19068730/how-do-i-change-the-autoindent-to-2-space-in-ipython-notebook
 

## TODO
 
TODO:
 - [x] Extract keypoints frame by frame to *.npy files.
 - [x] Label frames to defined classes.
 - [ ] Experiment with recurrent models to punch classification.
 - [ ] Add some advanced features, e.g. angles, distances e.t.c.
 - [ ] Get more videos.
 
 Links:  

Awesome Action Recognition
https://github.com/jinwchoi/awesome-action-recognition  

Anton Broilovskiy, Makarov I. Human Action Recognition for Boxing Training Simulator. AIST 2020
