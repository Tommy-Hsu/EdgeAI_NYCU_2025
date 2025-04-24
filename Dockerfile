# Dockerfile for PCT environment

FROM python:3.11

RUN apt-get update && \
    pip install --upgrade pip && \
    pip install matplotlib && \
    pip install tqdm  && \
    pip install torchprofile && \
    pip install torch-pruning --upgrade && \
    pip install ipykernel && \
    pip install huggingface-hub[cli] && \
    pip install transformers==4.50.3 && \
    pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124 && \ 
    pip install torchvision torchaudio && \
    pip install timm==1.0.15 && \
    pip install datasets==3.5.0 && \
    pip install accelerate==1.6.0 && \
    pip install gemlite==0.4.4 && \
    pip install hqq==0.2.5 && \
    pip install triton==3.2.0