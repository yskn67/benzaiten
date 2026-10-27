# Use same python version as google colaboratory
# FROM python:3.10.12
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    ca-certificates fluidsynth ffmpeg musescore3 build-essential libssl-dev libffi-dev \
    python3 python3-dev python3-pip git wget unzip \
    && rm -rf /var/lib/apt/lists/*

# pip・Requests・gRPCの独自CAも、更新したOSのCAへ揃える。
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    GRPC_DEFAULT_SSL_ROOTS_FILE_PATH=/etc/ssl/certs/ca-certificates.crt
# build-essential libasound2-dev libjack-dev portaudio19-dev sox

# Copy the requirements.txt and run pip install.
COPY ./requirements.txt ./
COPY ./requirements.pytorch.txt ./
RUN pip install -U "pip>=26.2.2"
RUN pip install -r requirements.pytorch.txt
RUN pip install -r requirements.txt "certifi>=2026.10.19"
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
