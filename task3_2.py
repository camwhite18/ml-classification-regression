from typing import List

import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score, f1_score, r2_score, \
    mean_absolute_error, mean_squared_error
from sklearn.svm import SVC


def evaluate_classification_model(model: object, x_test: np.ndarray, y_test: np.ndarray, display_labels: List[str],
                                  plot: bool = True) -> None:
    """
   This function is used to evaluate a classification model on a test set. It calculates and prints out the accuracy,
   precision, recall, and F1-score of the model. If plot is set to True, it also displays a confusion matrix.

   Parameters:
       model (object): The machine learning model to evaluate. It should be a trained model.
       x_test (np.ndarray): The test features. It should be a 2D numpy array, where each row is a sample and each
                        column is a feature.
       y_test (np.ndarray): The true labels for the test set. It should be a 1D numpy array with the same length as
                        the number of samples in x_test.
       display_labels (List[str]): The labels to display on the confusion matrix. It should be a list of strings, where
                        the strings are the names of the classes in the dataset.
       plot (bool, optional): Whether to plot a confusion matrix. The default is True.

   Returns:
       None
   """
    y_pred = model.predict(x_test)
    print(f'Accuracy: {model.score(x_test, y_test)}')
    print(f'Precision: {precision_score(y_test, y_pred, average="weighted")}')
    print(f'Recall: {recall_score(y_test, y_pred, average="weighted")}')
    print(f'F1-score: {f1_score(y_test, y_pred, average="weighted")}')
    if plot:
        ConfusionMatrixDisplay.from_predictions(y_pred, y_test, display_labels=display_labels,
                                                cmap=plt.cm.Blues)


def evaluate_regression_model(model: object, x_test: np.ndarray, y_test: np.ndarray, plot: bool = True) -> None:
    """
    This function is used to evaluate a regression model on a test set. It calculates and prints out the R^2 score,
    Mean Absolute Error, and Root Mean Squared Error of the model. If plot is set to True, it also plots a scatter plot
    of actual vs predicted values.

    Parameters:
        model (object): The machine learning model to evaluate. It should be a trained model.
        x_test (np.ndarray): The test features. It should be a 2D numpy array, where each row is a sample and each
                            column is a feature.
        y_test (np.ndarray): The true labels for the test set. It should be a 1D numpy array with the same length as
                            the number of samples in x_test.
        plot (bool, optional): Whether to plot actual vs predicted values. The default is True.

    Returns:
        None
    """
    # Use the model to make predictions on the test set
    y_pred = model.predict(x_test)
    # Print out the R^2 score, Mean Absolute Error, and Root Mean Squared Error
    print(f'R^2 Score: {r2_score(y_test, y_pred)}')
    print(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}')
    print(f'Root Mean Squared Error: {mean_squared_error(y_test, y_pred, squared=False)}')
    # Plot actual vs predicted values
    if plot:
        max_val = max(np.max(y_test), np.max(y_pred))
        plt.scatter(y_test, y_pred)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.xlim([0, max_val])
        plt.ylim([0, max_val])
        plt.title('Actual vs Predicted Values')
        plt.plot([0, max_val], [0, max_val], 'g--')
        plt.show()


def create_tuned_star_svc() -> SVC:
    """
    This function creates a Support Vector Machine classifier with its hyperparameters tuned on the stellar
    classification dataset. The hyperparameters are tuned using a grid search with 5-fold cross validation. The
    hyperparameters are tuned as follows:
        - C: 1000
        - gamma: 0.1
        - kernel: rbf

    Returns:
        SVC: The tuned Support Vector Machine classifier.
    """
    return SVC(kernel='rbf', C=1000, gamma=0.1)


def create_tuned_gwp_rf() -> RandomForestRegressor:
    """
    This function creates a Random Forest regressor with its hyperparameters tuned on the garment worker productivity
    dataset. The hyperparameters are tuned using a grid search with 5-fold cross validation. The hyperparameters are
    tuned as follows:
        - n_estimators: 50
        - max_depth: 20
        - min_samples_leaf: 4
        - min_samples_split: 2
        - bootstrap: True

    Returns:
        RandomForestRegressor: The tuned Random Forest regressor.
    """
    return RandomForestRegressor(n_estimators=50, max_depth=20, min_samples_leaf=4, min_samples_split=2,
                                 bootstrap=True)
