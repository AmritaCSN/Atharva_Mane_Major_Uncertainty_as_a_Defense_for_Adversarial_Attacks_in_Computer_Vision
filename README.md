# Uncertainty as a Defense for Adversarial Attacks in Computer Vision

## ABOUT
This repository contains code implementation of Minimum Prediction Deviation (MPD), an uncertainty metric, as a defense mechanism against adversarial attacks in computer vision systems.

### Architechture/Flow Daigram:
<p align="center">
  <img src="Architechture Diagram.png?raw=true" alt="Flow Diagram">
  <br />
  <em>Flow Diagram</em>
</p>


### HOWTO
- AEFinder.py - Use this to find the best parameters for the Adversarial Attacks.
- AP_Finder.py - Use this to find the best parameters for Adversarial Patch attack only.
- AE_Generator.py - Use this to generate the AEs based on the parameters found.
- MPD.py - Use this to create your MPD Detector (Pass the Ensemble to this module).
- MPD_Thres.py - This code will calculate the best threshold and evaluate the samples using MPD.
- MNIST_100_DNN - Use this code to generate the Ensemble of DNNs for MNIST Dataset (count is set to 100).
- GTSRB_100_DNN - Use this code to generate the Ensemble of DNNs for GTSRB Dataset (count is set to 100).

## Overview

Computer vision has become a pervasive tool, but its susceptibility to adversarial attacks poses a significant challenge. Adversarial examples, small input changes designed to deceive machine learning models, can lead to incorrect predictions or decisions, known as evasion attacks. In this project, we evaluate the effectiveness of Minimum Prediction Deviation (MPD) as an uncertainty defense in the field of computer vision.

## Dependencies

Make sure you have the following dependencies installed:

- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [ART (Adversarial Robustness Toolbox)](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
- [Scikit-learn](https://scikit-learn.org/stable/)
  
You can install them using:

```bash
pip install torch numpy adversarial-robustness-toolbox scikit-learn
```
## Dataset

### MNIST

- The MNIST database of handwritten digits has a training set of 60,000 examples and a test set of 10,000 examples.
- Each image is a 28x28 pixel grayscale representation of a handwritten digit (0 through 9).
- This dataset is widely used for training and evaluating machine learning models in image recognition, providing a diverse range of writing styles and digit variations.
### GTSRB
- GTSRB dataset is a benchmark collection of traffic sign images used for training and evaluating machine learning models in traffic sign recognition.
- It consists of images of traffic signs from various real-world scenarios, captured under different lighting conditions, weather conditions, and viewpoints.
- Each image is a 32x32 pixel RGB representation of a traffic sign (43 classes).

## Adversarial Robustness Toolbox (ART)

- Adversarial Robustness Toolbox (ART) is a Python library for Machine Learning Security.
- ART provides tools to evaluate, defend, certify, and verify machine learning models and applications against adversarial threats.
- It supports all popular machine learning frameworks (TensorFlow, Keras, PyTorch, MXNet, scikit-learn, XGBoost, LightGBM, CatBoost, GPy, etc.), all data types (images, tables, audio, video, etc.), and machine learning tasks (classification, object detection, generation, certification, etc.).

## Adversarial Attacks

Different adversarial attacks used in the code:

- Basic Iterative Method
- Projected Gradient Descent
- Auto Projected Gradient Descent
- Carlini and Wagner Attack
- Adversarial Patch

## Minimum Prediction Deviation (MPD)

MPD is a metric that measures the uncertainty of a machine learning model's prediction for a single sample. It relies on the distribution of probabilistic predictions generated by an ensemble of bootstrapped estimators. MPD quantifies inconsistencies within the model's predictions in a meaningful manner.


## Flow of the Code

1. **Import the Dataset:**
   - Load the dataset consisting of a training set and a test set using the Dataloaders from Pytorch.

2. **Create a Neural Net Model:**
   - Create a neural network model for image classification using PyTorch and train that model.

3. **Create an ART Classifier:**
   - Create an ART classifier and pass the PyTorch model to it.

4. **Evaluate on the Test Set:**
   - Evaluate the model's accuracy on the test set.

5. **Generate Adversarial Examples:**
   - Use various adversarial attacks (e.g., Basic Iterative Method, Projected Gradient Descent, Auto PGD, Carlini and Wagner Attack, Adversarial Patch) to generate adversarial examples for the entire test set.
   - Firstly, use the AEFinder module to find the best parameters for all the attacks (AP_Finder for Adversarial Patch attack).
   - Use the AEGenerator module to generate the AEs for all attacks and make sure to input all the parameters found during AEFinder code.
     
6. **Evaluate the Pytorch Classifier on the AE Set:**
   - Evaluate the model's accuracy on the AE test set.
  
7. **Generate Ensembles of DNNs:**
   - Generate 'n' number of DNNs using the same Pytorch Model while bootstrapping the data and save them in a folder.
   - Utilize the <dataset_100_DNN>.py code to generate the DNNs.
   - The code can be modified with the desired architecture and dataset to create the ensemble of DNNs.
     
8. **Create the MPD Detector:**
   - Pass the Ensemble of DNNs to the MPD.py file to create the MPD Detector to obtain the MPD Scores for the sample images.
   - Additionally, you can evaluate the test sets of Clean and AE samples to get a graph of the MPD scores on both type of samples.
  
9. **Create the Testbed:**
   - Generate the testbed for evaluation.
   - The testbed consists of total 10000 samples with 75% clean and 25% AE samples.

10. **Utilize the MPD Threshold Calculator:**
   - Pass the Base Classifier, MPD Detector and the AE samples and the clean 75% samples.
   - The MPD_thres module will generate testbed by stratifying the 25% samples from the entire AE test set.
   - Using the testbed, a threshold is determined for classifying the samples as Clean or AE sample.
     
11. **Employ MPD for Detection:**
   - Utilize Minimum Prediction Deviation (MPD) as an uncertainty metric to detect adversarial examples generated by the attacks.
   - The samples above the threshold get classified as AEs and below the threshold get classified as Clean samples.
   - Evaluate the testbed containing 10000 samples.
   - Additionally, you can evaluate the false positives on the clean model to gather some extra observations. 

