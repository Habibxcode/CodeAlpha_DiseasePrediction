# Disease Prediction from Medical Data - CodeAlpha

## Project Overview
This repository contains the implementation of Task 4: Disease Prediction from Medical Data, completed as part of the CodeAlpha Machine Learning Internship[cite: 1, 2]. The primary objective of this project is to build a robust machine learning model capable of predicting the possibility of disease based on structured patient diagnostic data[cite: 2]. Specifically, this project focuses on classifying breast cancer tumors as either malignant or benign.

## Dataset Information
The project utilizes the Breast Cancer Wisconsin (Diagnostic) dataset, which is a standard benchmark dataset provided by the UCI Machine Learning Repository and accessible directly through the `scikit-learn` library[cite: 2]. 
* **Instances:** 569 patient records.
* **Features:** 30 numerical features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass[cite: 2]. These features describe characteristics of the cell nuclei present in the image, such as radius, texture, perimeter, area, and smoothness.
* **Target Classes:** 0 (Malignant) and 1 (Benign).

## Methodology and Architecture
This system follows a standard supervised machine learning pipeline for structured data[cite: 2]:
1. **Data Ingestion:** The dataset is loaded and converted into a pandas DataFrame for structured manipulation.
2. **Data Splitting:** The data is split into training (80%) and testing (20%) sets to ensure the model is evaluated on unseen data.
3. **Data Preprocessing:** A `StandardScaler` is applied to normalize the feature variations. This ensures that features with larger numeric ranges do not disproportionately influence the model.
4. **Model Selection:** A Random Forest Classifier is used for this task[cite: 2]. Random Forest is an ensemble learning method that operates by constructing a multitude of decision trees during training and outputting the mode of the classes for predictions.
5. **Model Evaluation:** The model is evaluated using Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix to provide a comprehensive view of its diagnostic performance[cite: 1].

## Prerequisites and Installation
To run this project locally, ensure you have Python installed on your system. 

1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/CodeAlpha_DiseasePrediction.git](https://github.com/YourUsername/CodeAlpha_DiseasePrediction.git)
   cd CodeAlpha_DiseasePrediction