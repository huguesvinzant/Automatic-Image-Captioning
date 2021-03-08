# Automatic Image Captioning

## Project Overview

This repository contains the files for the second project of the Udacity Computer Vision Expert Nanodegree. It combine knowledge of Computer Vision Techniques and Deep learning Architectures to create a neural network architecture to automatically generate captions from images.

## Example
<p align="center"> <img src="images/image-captioning.png" align="middle" width="100%"> </p>

## Project Structure
The project is structured as a series of Jupyter notebooks that are designed to be completed in sequential order:

__Notebook 0__ : Loading and Visualizing the The Microsoft Common Objects in Context (MS COCO) dataset

__Notebook 1__ : Pre-process data from the COCO dataset. Design a CNN-RNN model for automatic image captions generation

__Notebook 2__ : Train the CNN-RNN model

__Notebook 4__ : Use the trained model to generate captions for images in the test dataset

`models.py` : Definition of the CNN-RNN model to be used for automatic image captions generation

### Local Environment Instructions

1. Clone the repository.
	```
	git clone https://github.com/nalbert9/Facial-Keypoint-Detection.git
	```
2. Create (and activate) a new Anaconda environment (Python 3.6).

	- __Linux__ or __Mac__: 
	```
	conda create -n cv-nd python=3.6
	source activate cv-nd
	```
	- __Windows__: 
	```
	conda create --name cv-nd python=3.6
	activate cv-nd
	```

3. Install PyTorch and torchvision; this should install the latest version of PyTorch;
```
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```
6. Install a few required pip packages, which are specified in the requirements text file (including OpenCV).
```
pip install -r requirements.txt
```
