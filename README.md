# s2-superresolution on UP42
## Introduction

This is a simple instructional proof-of-concept of using [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network) 
algorithm to drive higher resolution images from the existing lower resolution images to provide 
a very simple demo of what is possible to do using UP42.

The goal of this project is to guide you through setting UP42 in your
geospatial pipeline. It shows how easy it is to setup a CNN
algorithm implemented in [TensorFlow](https://tensorflow.org) to
perform CNN.

## Block description

This is the
[block](https://docs.up42.com/getting-started/core-concepts.html#blocks)
description in terms of the UP42 core concepts.

* Block type: processing
* Supported input types:
  * [SENTINEL2_L1C](https://specs.up42.com/v1/blocks/schema.json) 
  (any geo-referenced [GeoTIFF](https://en.wikipedia.org/wiki/GeoTIFF))
* Provider: [UP42](https://up42.com)
* Tags: machine learning, data processing, analytics

## Requirements

 1. [git](https://git-scm.com/).
 2. [docker engine](https://docs.docker.com/engine/).
 3. [UP42](https://up42.com) account credentials.
 4. [Python](https://python.org) 3.5 or later.
 5. Required Python packages as specified in
    `blocks/s2-superresolution/requirements.txt`.

## Usage

### Local development HOWTO

Clone the repository in a given `<directory>`:

```bash
git clone https://github.com/up42/s2-superresolutiongit <directory>
``` 

then do `cd <directory>`.
#### Install the required libraries
First create a virtual environment either by using [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) 
or [virtualenv](https://virtualenv.pypa.io/en/latest/).
In the case of using the virtualenvwrapper do:

```mkvirtualenv --python=$(which python3.7) up42-supres```

In the case of using the virtualenv do:

````
virtualenv -p $(which python3.7) up42-supres
````

After creating a virtual environment and activating it, all the necessary libraries can be installed on this environment by doing:

```bash
./s2-superresolution/blocks/s2_superresolution/setup.sh
```
#### Create the super resolution image using the trained network

The trained network can be used directly on downloaded Sentinel-2 tiles. for more details, see in the s2_tiles_supres.py file.


The code use the .xml file of the uzipped S2 tile. The output image will be saved with a `.tif` extension which is easily readable by QGIS.
You need to create a `/tmp/output/` so that the output image will be written into this directory.
If you want to also copy the original high resolution (10m bands) you can do so, with tuning the parameter `copy_original_bands` in the Up42Manifest.json and choose default to be `yes`.
The source code also predict the lowest resolution bands (60m) by default, so that the output image will include high resolution (10m) for all the exiting bands
within 20m and 60m resolutions.


#### Run the tests

This project uses [pytest](https://docs.pytest.org/en/latest/) for testing. To run
the tests first create a `/tmp/input/` directory and place your image containing the `.SAFE` file in this directory. 
Therefore the input image will be read from the `/tmp/input/` directory. Then do as following:

```bash
./s2-superresolution/blocks/s2-superresolution/test.sh
```

from the repository top directory.

#### Build the processing block Docker image 

To build the Docker image for local testing and/or publishing on the UP42
platform you can run the following shell commands from the repository
top directory:

```bash
cd blocks/s2-superresolution/
# Build the image.
docker build -t s2-superresolution/-f Dockerfile . --build-arg manifest="$(cat UP42Manifest.json)"
# Go back to the top directory.
cd -
```

#### Run the processsing block 

 * Make sure you have created the block input `/tmp/input` and output directories `/tmp/output`.
 * Copy the input data form the unzipped s2 file (along with the
   [GeoJSON](https://en.wikipedia.org/wiki/GeoJSON) file called
   `data.json`) to `/tmp/input`.
 * Build the docker image as outlined above.
 * Run the following command: 
 
```bash
docker run --mount type=bind,src=/tmp/output,dst=/tmp/output --mount type=bind,src=/tmp/input,dst=/tmp/input s2-superresolution:latest
``` 

This [bind mounts](https://docs.docker.com/storage/bind-mounts/) the
host and container `/tmp/input` and `/tmp/output` directories into the
**input** and **output** directories respectively. If you wish you can
set it to some other directory that is convenient to you.

### Publish the block to UP42

#### Authenticate into the UP42 registry 

Login into the UP42 [Docker image registry](https://docs.docker.com/registry/) 
with your UP42 user ID (`<user_id>`) and password:

```bash
docker login -u <user_id> registry.up42.com
``` 
#### Push the block to the registry

Push your block as a Docker image to the UP42 registry like this: 

```bash
docker push registry.up42.com/<user_id>/s2-superresolution:latest
```

Learn more about creating and publishing blocks by reading our
[documentation](https://docs.up42.com/getting-started/first-block.html#).

### Further resources

 * [Getting started with UP42](https://docs.up42.com/getting-started/index.html)
 * [Creating a block](https://docs.up42.com/getting-started/first-block.html)
 * [Setting up the development environment](https://docs.up42.com/getting-started/dev-setup.html)
 * [Block specifications](https://docs.up42.com/specifications/index.html)
 * [Block examples](https://docs.up42.com/examples/index.html)
 * [Tensorflow](https://www.tensorflow.org/)

### Support
  
 1. Open an issue here.
 2. Reach out to us on
      [gitter](https://gitter.im/up42-com/community).
 3. Mail us [support@up42.com](mailto:support@up42.com).

 


