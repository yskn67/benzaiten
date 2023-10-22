# Use same python version as google colaboratory
# FROM python:3.10.12
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt update
RUN apt install -y fluidsynth ffmpeg musescore3 build-essential libssl-dev libffi-dev python3 python3-dev python3-pip git wget unzip
# build-essential libasound2-dev libjack-dev portaudio19-dev sox

# Copy the requirements.txt and run pip install.
COPY ./requirements.txt ./
COPY ./requirements.pytorch.txt ./
RUN pip install -U pip
RUN pip install -r requirements.pytorch.txt
RUN pip install -r requirements.txt
RUN rm requirements.txt
RUN rm requirements.pytorch.txt

# Setting jupyter_notebook_config
RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.notebook_dir = '/workspace/notebook'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.password = ''" >> /root/.jupyter/jupyter_notebook_config.py

# workspace
RUN mkdir /workspace
WORKDIR /workspace/
