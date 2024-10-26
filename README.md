# Project Overview

This project is structured to efficiently handle model training and AUC comparison. The main components of the project include:

## Files

- **utils.py**: Contains helper functions for various tasks such as:
  - Data standardization
  - AUC score calculation
  - Saving predictions to CSV
  - AUC score comparison

- **models.py**: Includes functions for training different models:
  - `train_knn`: Trains a K-Nearest Neighbors (KNN) classifier
  - `train_logistic_regression`: Trains a Logistic Regression model
  - `train_svm`: Trains a Support Vector Machine (SVM) model with a polynomial kernel
  - `train_neural_network`: Trains a Multilayer Perceptron (Neural Network) model
  - `train_random_forest`: Trains a Random Forest classifier

- **CP1_model_test.ipynb**: Jupyter Notebook containing tests and implementation details for each of the models. The notebook demonstrates the usage of `utils.py` and `models.py` to train and evaluate models.

## Key Functions

- **AUC Comparison Function**: `compare_auc`  
  This function compares the AUC scores between two models for each dataset and returns the differences.  
  You can see an example of its usage in the `CP1_model_test.ipynb` notebook.
