# s2-superresolution on UP42
## Introduction

This is a state of the are processing block using a [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network) 
algorithm to derive higher resolution images from existing lower resolution images using Sentinel-2 datasets as input.
The code is adapted from https://github.com/lanha/DSen2, our thanks go to the authors of the original code base and the 
corresponding paper.

Another goal of this project is to help users setting up their [TensorFlow](https://tensorflow.org) based algorithms on
[UP42](https://up42.com).

## Block description

This is the
[block](https://docs.up42.com/getting-started/core-concepts.html#blocks)
description in terms of the UP42 core concepts.

* Block type: processing
* Supported input types:
  * [SENTINEL2_L1C](https://docs.up42.com/up42-blocks/sobloo-s2-l1c.html)   
* Output type: AOIClipped (geo-referenced [GeoTIFF](https://en.wikipedia.org/wiki/GeoTIFF))
* Provider: [UP42](https://up42.com)
* Tags: machine learning, deep learning, data processing, analytics

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
git clone https://github.com/up42/s2-superresolution.git <directory>
``` 

then do `cd <directory>`.
#### Install the required libraries
First create a virtual environment either by using [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) 
or [virtualenv](https://virtualenv.pypa.io/en/latest/).
In the case of using virtualenvwrapper do:

```mkvirtualenv --python=$(which python3.7) up42-supres```

In the case of using virtualenv do:

````
virtualenv -p $(which python3.7) up42-supres
````

After creating a virtual environment and activating it, all the necessary libraries can be installed on this environment by doing:

```bash
cd s2-superresolution/blocks/s2_superresolution/
./setup.sh
```
#### Create the super resolution image using the trained network

The trained network can be used directly on downloaded Sentinel-2 tiles. For more details, see in the s2_tiles_supres.py file.

The code use the .xml file of the unzipped S2 tile. The output image will be saved with a `.tif` extension which is
easily readable by GIS packages such as QGIS (https://www.qgis.org/).
You need to create a `/tmp/output/` folder so that the output image can be written into this directory.
If you want to also copy the original high resolution (10m bands) you can do so, with tuning the parameter
`copy_original_bands` in the Up42Manifest.json and choose default to be `yes`.
The source code also predict the lowest resolution bands (60m) by default, so that the output image will include high
resolution (10m) for all the exiting bands within 20m and 60m resolutions.


### Run the tests

This project uses [pytest](https://docs.pytest.org/en/latest/) for testing. It is necessary to provde a Sentinel-2
dataset in SAFE format.  
Please note that the input image should be the outcome of the `Sentinel-2 L1C MSI Full Scenes` data block which contains a
``data.json``. This ``data.json`` file is needed for running the test file correctly.
To run the tests, first create a `/tmp/input/` directory and place the result of the mentioned data block ( which
includes a `.SAFE` file) in this directory. 
The output image will be written to the `/tmp/output/` directory. Finally, to run the test do as following:

```bash
./test.sh
```

### Build and run the docker image locally

To build the Docker image for local using you can run the following shell command from the repository
that contains the Dockerfile: 

```bash
cd s2-superresolution/blocks/s2-superresolution/
# Build the image.
docker build -t s2-superresolution -f Dockerfile . 

```
In the next step you can use the params.json file to define whether you want to work with the whole image or 
a subset of the image by modifying the ``roi_x_y`` or ``roi_lon_lat`` (which is a `list` of coordinates).
You can also choose whether you want to keep the original spectral bands of 10m resolution of the input image or not. 
Please note that you can add or remove ``roi_x_y`` or ``roi_lon_lat`` based on your preferences.

An example of params.json file is shown below:

``
{
  "roi_x_y": [5000,5000,5500,5500],
  "copy_original_bands": false
}
``

#### Run the processing block 

 * Make sure you have created the block input `/tmp/input` and output directories `/tmp/output`.
 * Copy the input data form the unzipped s2 file (along with the
   [GeoJSON](https://en.wikipedia.org/wiki/GeoJSON) file called
   `data.json`) to `/tmp/input`.
 * Build the docker image as outlined above.
 * Run the following command: 
 
```
 docker run -u -e UP42_TASK_PARAMETERS="$(cat params.json)" --mount type=bind,src=/tmp/output,dst=/tmp/output --mount type=bind,src=/tmp/input,dst=/tmp/input superresolution:latest
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

To build the Docker image for publishing on the UP42
platform you can run the following shell commands from the repository
that contains the Dockerfile:

```bash
cd s2-superresolution/blocks/s2-superresolution/
# Build the image.
docker build . \
     -t registry.up42.com/some-example-user-id/s2-superresolution:latest \
     --build-arg manifest="$(<UP42Manifest.json)"
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

 


