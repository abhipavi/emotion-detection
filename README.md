# Leveraging Transfer Learning to Enhance Real-Time Emotion Recognition

## Introduction
Emotions are a universal aspect of being human, regardless of age. They manifest as bodily sensations and behaviors, reflecting personal significance. Computers understanding non-verbal cues can have various applications, like assessing student interest in education, aiding in psychiatry and autism treatment, detecting lies in interrogations, and even enhancing ATM security.

Convolutional Neural Networks (CNNs) are powerful tools in image analysis, adept at extracting features from images. However, training CNNs from scratch is time-consuming and requires extensive datasets. Transfer learning addresses this by using pre-trained networks on large datasets, fine-tuning them for specific tasks. This approach, utilizing networks like ResNet and GoogleNet, allows for real-time emotion detection.

## Dataset
Two datasets were considered for training the model. The AffectNet dataset, the largest of its kind, comprises 450,000 annotated images gathered from search engines using emotional keywords. It employs both categorical and valence-arousal scale models to measure emotions, offering a comprehensive view of emotional responses. The FER+ dataset, an improved version of FER, contains 28,709 images with seven emotion labels. Tags were refined using crowd-sourcing, enhancing tag accuracy and inter-rater agreement to over 80%. For this project, the FER+ dataset is chosen as the primary dataset. The FER+ dataset contains a large collection of facial images with labeled emotions, making it suitable for training emotion recognition models. The dataset is publicly available and licensed under the MIT license.

## Ethical Use of Data
The primary dataset for this project is the FER+ dataset, which is publicly available and licensed under the MIT license. It contains images of human participants with labeled emotions. To address the need for real-time emotion detection without human participants, I will evaluate the program myself and use a publicly available dataset for testing. This approach eliminates the need for additional human participants, aligning with ethical guidelines.

## Model Training
The FER+ dataset undergoes preprocessing before training with selected pre-trained models, including VGG16, VGG19, InceptionV3, and Xception. The number of unlocked layers for training is determined through experimentation, typically unlocking at least three convolution layers. The final model architecture includes three dense layers, with the last layer being a softmax layer. State-of-the-art models achieve approximately 73% accuracy, so an accuracy of 55-65% is expected. Categorical cross-entropy is used as the loss function, and the number of epochs is determined through trial and error to prevent overfitting or underfitting. 

## Real Time Detection
Real-time emotion detection involves two steps: facial recognition and emotion recognition. Facial detection is performed using OpenCV and Haar classifiers, which detect faces based on features like edges and lines. Haar cascade employs a cascading window to classify objects. Specific Haar features for facial detection are stored in an XML file and imported via OpenCV.

Detected faces undergo preprocessing to adjust size and convert to black and white before passing through the fine-tuned CNN model. The predicted emotion is displayed on screen. The software is developed in Python using Visual Studio Code on a local machine.

## Running the project
To run the project, follow these steps:
* Download the FER2013 dataset from Kaggle(https://www.kaggle.com/datasets/msambare/fer2013).
* Run the model_training_code.ipynb notebook to train the CNN models and export them in h5 and JSON formats.
* Import the relevant h5 and JSON files into the real_time_detection.ipynb notebook.
* Run all cells in the real_time_detection.ipynb notebook to perform real-time emotion recognition.


