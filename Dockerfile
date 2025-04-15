# Dockerfile for PCT environment

FROM python:3.11

RUN apt-get update && \
    pip install --upgrade pip && \
    pip install matplotlib && \
    pip install tqdm  && \
    pip install torchprofile && \
    pip install torch-pruning --upgrade && \
    pip install ipykernel