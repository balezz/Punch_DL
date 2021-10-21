# Deep learning models for punch classification in boxing and karate  

This repository contains data and code with our punch classification experiments. 
We are using acceleration sensors measurements and video frames for punch class recognition.

`DOI_10.1109ACCESS.2021.3118038` - data and code for reproducing classification metric results for article 
'Recognition punches in karate using acceleration sensors and convolution neural networks.  
 
`MoveNetExtractKeypoints.ipynb` - keypoints extraction for punch videos.  
`SimpleRNN.ipynb` - vanilla RNN as baseline.  

Starting videos available [here](https://drive.google.com/drive/folders/1UwZPZ7sqkmQrqbCP1ypquv2UHWkk0bj-?usp=sharing).  
On each video two man (id0 and id1) have done 10 weak (5 left and 5 right hand) and 10 strong punches. 
Total 240 punches in dataset v0.1 We start with only punch class prediction, no power estimation.  

Box punches classes: 
0) no punch,
1) jab (jab left), 
2) cross (jab right), 
3) left hook, 
4) right hook, 
5) left uppercut, 
6) right uppercut.   

Build docker image to reproduce results: `docker build -t punch_dl:v1 .`   
 
TODO:
 - [x] Extract keypoints frame by frame to *.npy files.
 - [ ] Label frames to defined classes.
 - [ ] Experiment with recurrent models to punch classification.
 - [ ] Add some advanced features, e.g. angles, distances e.t.c.
 
