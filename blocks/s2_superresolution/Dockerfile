### Dockerfile to build the UP42 usage tutorial for
### superresolution.

# Use one of the official Tensorflow Docker images as base.
FROM tensorflow/tensorflow:1.13.1-py3

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