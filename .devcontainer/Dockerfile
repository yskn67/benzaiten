# Use same python version as google colaboratory
FROM python:3.10.12

RUN apt update
RUN apt install -y fluidsynth ffmpeg musescore3
# build-essential libasound2-dev libjack-dev portaudio19-dev sox

# Copy the requirements.txt and run pip install.
COPY ./requirements.txt ./
RUN pip install -U pip
RUN pip install -r requirements.txt

# Setting jupyter_notebook_config
RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.notebook_dir = '/workspace/notebook'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.password = ''" >> /root/.jupyter/jupyter_notebook_config.py