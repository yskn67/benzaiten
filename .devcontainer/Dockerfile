# Use same python version as google colaboratory
FROM python:3.10.12

RUN apt update

# Copy the requirements.txt and run pip install.
COPY ./requirements.txt ./
RUN pip install -U pip
RUN pip install -r requirements.txt