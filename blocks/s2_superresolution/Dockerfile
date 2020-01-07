### Dockerfile to build the UP42 superresolution block.

# Use one of the official Tensorflow Docker images as base.
FROM tensorflow/tensorflow:latest-gpu-py3
LABEL maintainer="NVIDIA CORPORATION <cudatools@nvidia.com>"

ENV CUDNN_VERSION 7.6.5.32
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn7=$CUDNN_VERSION-1+cuda10.0 \
&& \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

# FROM tensorflow/tensorflow:1.13.1-py3

# The manifest file contains metadata for correctly building and
# tagging the Docker image. This is a build time argument.
ARG manifest
LABEL "up42_manifest"=$manifest

# Working directory setup.
WORKDIR /block
COPY requirements.txt /block

# Install the Python requirements.
RUN pip install -r requirements.txt

# Copy the code into the container.
COPY src /block/src
COPY weights /block/weights

# Invoke run.py.
CMD ["python", "/block/src/predict/run.py"]