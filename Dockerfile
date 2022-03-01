FROM tensorflow/tensorflow:2.5.0-gpu-jupyter

RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install opencv-python imageio tensorflow-hub
RUN pip install -q git+https://github.com/tensorflow/docs

RUN pip install pandas seaborn keras-tuner

RUN apt install graphviz -y
RUN pip install pydot graphviz