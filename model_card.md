# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

– Person or organization developing model: Michael Schock
– Model date: 4/2/2026
– Model version: 1.0.0
– Model type: Logistic Regression
– Information: This is a Logistic Regression model using a simple scaler and one hot encoding, with no defined parameters.

## Intended Use

This model predicts whether income level exceeds $50K. It is intended as a simple example for a machine learning pipeline as part of the Udacity Machine Learning DevOps course. This model is not intended for actual use, but to serve as a educational project.

## Training Data

This model was trained on a subset of records from the Census Income dataset containing 32,562 records and 15 features. No parameters were specified, so the default 75/25 train/test split applies. 

## Evaluation Data

The model was evaluated against the test data consisting of 8,140 records split from the original dataset (25%).

## Metrics
Precision: 0.2379 | Recall: 1.0000 | F1: 0.3844

## Ethical Considerations
This model may reflect biases inherent in the training data, particularly regarding demographic features such as race and gender. These biases can lead to skewed predictions that may affect certain groups more than others.
## Caveats and Recommendations
This model may not generalize well to different populations or datasets. The model was created as part of an educational project centered around deployment of an ML model and not the training, accuracy, or viability of it. As such, it is advisable to keep these limitations in mind.