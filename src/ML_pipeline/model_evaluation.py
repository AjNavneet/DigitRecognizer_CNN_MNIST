import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Function to evaluate a machine learning model
def model_evaluate(model, X_test, y_test, verbose=0):
    """
    Evaluate the model on the test data.

    Args:
    model (object): The trained machine learning model.
    X_test (array-like): Test data features.
    y_test (array-like): True labels for the test data.
    verbose (int, optional): Verbosity mode for evaluation. Default is 0.

    Returns:
    list: A list of evaluation scores (e.g., loss and accuracy).
    """
    score = model.evaluate(X_test, y_test, verbose=verbose)
    return score

# Function to make predictions on test data
def predict(model, test_data):
    """
    Make predictions on test data using the given model.

    Args:
    model (object): The trained machine learning model.
    test_data (array-like): Test data for making predictions.

    Returns:
    array-like: Predicted labels.
    """
    predictions = np.argmax(model.predict(test_data), axis=-1)
    return predictions
 
# Function for generating a confusion matrix
def confusion_mat(test_data, predictions):
    """
    Generate a confusion matrix to evaluate classification performance.

    Args:
    test_data (array-like): True labels for the test data.
    predictions (array-like): Predicted labels.

    Returns:
    array-like: Confusion matrix.
    """
    conf_mat = confusion_matrix(test_data, predictions)
    return conf_mat

# Function for generating a classification report
def classification_mat(test_data, predictions):
    """
    Generate a classification report to evaluate classification performance.

    Args:
    test_data (array-like): True labels for the test data.
    predictions (array-like): Predicted labels.

    Returns:
    str: Classification report.
    """
    class_mat = classification_report(test_data, predictions)
    return class_mat

# Function to extract and store model performance metrics
def model_performance(model):
    """
    Extract and store model performance metrics during training and validation.

    Args:
    model (object): The trained machine learning model with training history.

    Returns:
    DataFrame: A pandas DataFrame containing training metrics (e.g., loss and accuracy).
    """
    training_metrics = pd.DataFrame(model.history.history)
    return training_metrics
