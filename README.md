
# Breast Cancer Detection

## Identify the problem

Breast cancer is the most common malignancy among women, accounting for nearly 1 in 3 cancers diagnosed among women in the United States, and it is the second leading cause of cancer death among women. Breast Cancer occurs as a result of abnormal growth of cells in the breast tissue, commonly referred to as a Tumour. A tumor does not mean cancer - tumours can be benign (not cancerous), pre-malignant (pre-cancerous), or malignant (cancerous). Tests such as MRI, mammogram, ultrasound and biopsy are commonly used to diagnose breast cancer performed.



## Expected Outcome

Given breast cancer results from breast fine needle aspiration (FNA) test (is a quick and simple procedure to perform, which removes some fluid or cells from a breast lesion or cyst (a lump, sore or swelling) with a fine needle like a blood sample needle). Since this build a model that can classify a breast cancer tumor using two training classification:

1= Malignant (Cancerous) - Present
0= Benign (Not Cancerous) -Absent


## Objective

Since the labels in the data are discrete, the predication falls into two categories, (i.e. Malignant or benign). In machine learning this is a classification problem.

Thus, the goal is to classify whether the breast cancer is benign or malignant and predict the recurrence and non-recurrence of malignant cases after a certain period. To achieve this we have used machine learning classification methods to fit a function that can predict the discrete class of new input.

## Identify Data Sources

The Breast Cancer datasets is available machine learning repository maintained by the University of California, Irvine. The dataset contains 569 samples of malignant and benign tumour cells.

The first two columns in the dataset store the unique ID numbers of the samples and the corresponding diagnosis (M=malignant, B=benign), respectively.
The columns 3-32 contain 30 real-value features that have been computed from digitized images of the cell nuclei, which can be used to build a model to predict whether a tumour is benign or malignant.

## Required Libraries

import numpy as np

import pandas as pd


import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

from sklearn import svm

import xgboost
## Accuracy

Logistic Regression Accuracy:

Logistic Regression achieved a notable accuracy of 95.90% in breast cancer detection. This algorithm is widely used for its simplicity and interpretability.
Support Vector Classifier Accuracy:

The Support Vector Classifier (SVC) demonstrated an even higher accuracy of 96.49%, showcasing its effectiveness in discriminating between malignant and benign tumors. This algorithm is particularly powerful for handling complex decision boundaries.
K-Nearest Neighbor Classifier Accuracy:

The K-Nearest Neighbor (KNN) Classifier also achieved an accuracy of 96.49%, matching the performance of the Support Vector Classifier. KNN is known for its simplicity and effectiveness in classification tasks.
XGBoost Accuracy:

XGBoost, a popular ensemble learning algorithm, delivered an accuracy of 91.23%. While slightly lower than some other algorithms, XGBoost is valuable for handling large datasets and capturing complex relationships.
Cross-Validation Mean Accuracy:

The mean accuracy value obtained through cross-validation across various models is 93.98%. Cross-validation provides a robust assessment of model performance and generalizability.