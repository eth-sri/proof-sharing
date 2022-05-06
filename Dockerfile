# Download base image ubuntu 18.04
FROM ubuntu:18.04

# LABEL about the custom image
LABEL version="1.0"
LABEL description="This is Docker image for the accompanying code for the 'Shared Certificates for Neural Network Verification' paper accepted at CAV 2022"

# Disable Prompt During Packages Installation
ARG DEBIAN_FRONTEND=noninteractive

# Update Ubuntu Software repository
RUN apt update

# Install required APT packages
RUN apt install -y wget vim git build-essential make && \
    rm -rf /var/lib/apt/lists/* && \
    apt clean

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p $HOME/miniconda

# Create conda environment
COPY environment.yml /
RUN bash -c "source ~/miniconda/bin/activate;conda env create -f /environment.yml"

# Automatically launch Miniconda
RUN echo "source ~/miniconda/bin/activate" >> "/root/.bashrc"
RUN echo "conda activate cav_proof_sharing" >> "/root/.bashrc"
RUN echo "cd /proof-sharing" >> "/root/.bashrc"
