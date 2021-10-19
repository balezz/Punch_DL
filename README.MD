# Deep learning models for punch classification in boxing and karate  

This repository contains data and code with our punch classification experiments. 
We are using acceleration sensors measurements and video frames for punch class recognition.

`DOI_10.1109ACCESS.2021.3118038` - data and code for reproducing classification metric results for article 
'Recognition punches in karate using acceleration sensors and convolution neural networks.  
 
`MoveNetExtractKeypoints.ipynb` - keypoints extraction for punch videos.  
`SimpleRNN.ipynb` - vanilla RNN as baseline.  

Starting videos availiable [here](https://drive.google.com/drive/folders/1UwZPZ7sqkmQrqbCP1ypquv2UHWkk0bj-?usp=sharing).  
Box punches classes: 
0) no punch,
1) jab, 
2) cross, 
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
 
