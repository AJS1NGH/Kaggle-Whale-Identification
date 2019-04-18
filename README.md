# Kaggle-Whale-Identification
Code repository for Kaggle Whale Identification Challenge
# What is the Challenge?
Classify 5005 types of whales by images of their flukes (tails). There are 5005 unique classes in total however one class called new_whale contains images of multiple whales which have not been classified in any of the other 5004 classes. About 60% of the classes contain only 1 image. Unlike the new_whale class each class contains images of whales of the same type. New_whale class contains roughly 40% of the images of the whole dataset.

# File Descriptions

**EdgeDetection.ipynb**
Takes in a single raw image and creates an image which only has the edges/contours of the original image. Basically highlights the shape and pattern of the tail of the whale.

**KGDataAugmentation.ipynb**
Created to tackle data imbalance, especially for classes with only 1 image. Code in this file applies 12 augmentations from Albumentations library to every single image in every class (except from the new_whale class since it has 10,000 images already). So it applies 12 augmentations to about 15,000 images resulting in a total of 180,000 images generated including the initial 15,000.

**KGEdgeImages.ipynb**
Created to apply edge detection to every single image (total 220k+). Uses 25 threads where each thread performs edge detection on approximately 8800 images.

**KGEnsemble14.ipynb**
Created to ensemble 14 neural networks and output a prediction for ~8000 test images. This approach was abandoned due to time limitations however parts of code were used to ensemble 3 networks.

**KGWClassExploration.ipynb**
Counts and shows number of images per class to help with detecting data imbalance.

**KGWImageAmplifier.ipynb**
Applies multiple augmentations to image using PIL library.

**KGWhaleClassActions.ipynb**
Further sorts already-sorted classes into folders by number of images per class within a certain range.

**KGWhaleFileSorting.ipynb**
Sorts raw image data into class folders by using the csv file provided in the competition.

**KaggleWhaleTrain.ipynb**
Main file. Trains neural network and writes Top 5 predicted classes of each image in test set to csv file for submission. 


