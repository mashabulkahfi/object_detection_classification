# Table of Contents
1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Setup and Installation](#setup-and-installation)
4. [Classification](#classification)
   - [Prepare the Dataset](#prepare-the-dataset)
   - [Train the Model](#train-the-model)
   - [Evaluate the Model](#evaluate-the-model)
5. [Object Detection](#object-detection)
   - [Prepare the Dataset](#prepare-the-dataset-1)
   - [Train the Model](#train-the-model-1)
   - [Evaluate the Model](#evaluate-the-model-1)
6. [Inference](#inference)

# Project Overview
This project focuses on two tasks:
1. **Image Classification**: Classifying vehicle images into categories such as "City Car," "Big Truck," etc.
2. **Object Detection**: Detecting and classifying objects in images using bounding boxes.

The project includes scripts for training, evaluation, and inference, along with utilities for dataset preparation.

# Directory Structure
root/
├── data/
│   ├── classification/
│   │   ├── raw
|   │   │   ├── vehicle-images-dataset
│   │   ├── train
│   │   ├── test
│   │   ├── valid
│   ├── detection/
│   │   ├── train
│   │   ├── test
│   │   ├── valid
├── logs/
│   ├── classification/
│   │   ├── best_model
│   │   ├── training_history
│   ├── detection/
│   │   ├── best_model
│   │   ├── training_history
├── model/
│   ├── classification/
│   │   ├── __init__.py
│   │   ├── eval.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── utils.py
|   ├── detection/
│   │   ├── __init__.py
│   │   ├── eval.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── utils.py
├── notebooks/
|   ├── ImageClassification.ipynb
|   ├── ImageDetection.ipynb
├── scripts/
│   ├── __init__.py
│   ├── download_data_classification.py
│   ├── download_data_detection.py
│   ├── utils.py

# Setup and Installation

## Prerequisites
- I run this on Python 3.10.16 version
- Install all requirements/libraries using

```bash 
pip install -r requirements.txt
```


1. Classification 
```markdown```
### Prepare the dataset 

For prepare the dataset, download this dataset manually: https://www.kaggle.com/datasets/lyensoetanto/vehicle-images-dataset

and put the unzip file under 
├── data/
│   ├── classification/
│   │   ├── raw
|   │   │   ├── vehicle-images-dataset

so it would be like 
├── data/
│   ├── classification/
│   │   ├── raw
|   │   │   ├── vehicle-images-dataset
│   |   │   │   ├── Big Truck
│   |   │   │   ├── City Car 
│   |   │   │   ├── ...

After that, you can run this command: (make sure you run in the root folder project) to prepare the dataset

```bash 
python3 -m scripts.download_data_classification 
```

```markdown```
### Training the model

Run this command, and you can change the number of epoch arguments.

```bash
python3 -m model.classification.train --num_epochs 10
```

After you run it will saving the best model and training logs


├── logs/
│   ├── classification/
│   │   ├── best_model
|   │   │   ├── best_model_weights.pth
│   │   ├── training_history
|   │   │   ├── training_logs.txt
|   │   │   ├── loss_history.png
|   │   │   ├── metric_history.png

```markdown```
### Eval the model using test dataset

Run this command.

```bash
python3 -m model.classification.eval
```

Example output:
Model weights loaded successfully from logs/classification/best_model/best_model_weights.pth
Test Loss: 0.2578
Test Accuracy: 91.12%

2. Object Detection
```markdown```
### Prepare the dataset 
To download and prepare dataset, you can directly run the below command. But before you run, you need to find/generate your roboflow api key in roboflow web.

Run this command
```bash
python3 -m scripts.download_data_detection --api_roboflow_key "<YOUR_GENERATED_ROBOFLOW_KEY>"
```

After you run that command, the directory would be like:
├── data/
│   ├── ...
│   ├── detection/
│   │   ├── train
│   │   ├── test
│   │   ├── valid

### Run the training model
To run the detection model you can run this command: you can also add another arguments --num_epochs

```bash
python3 -m model.detection.train --num_epochs 10
```

After you run it will saving the best model and training logs

├── logs/
│   ├── .../
│   ├── detection/
│   │   ├── best_model
|   │   │   ├── best_model_weights.pth
│   │   ├── training_history
|   │   │   ├── training_logs.txt
|   │   │   ├── loss_history.png


### Eval the model using test set

```bash
python3 -m model.detection.eval
```

This will show you the loss value that got from test set.

The example output:
Model weights loaded successfully from /content/logs/detection/best_model/best_model_weights.pth
Test Loss: 0.09089820507793911

3. Inference
You can do inference by run this command:

```bash
python3 -m inference --file_name "example_input.jpeg"
```

It will generate a new image that contain the box based on the detection model and the label prediction based on the image classification.