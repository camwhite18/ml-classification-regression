from typing import List

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ttest_rel
from sklearn.base import is_classifier, BaseEstimator
from sklearn.cluster import KMeans
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, recall_score, precision_score, r2_score, \
    mean_absolute_error, mean_squared_error, accuracy_score, silhouette_score, davies_bouldin_score, \
    calinski_harabasz_score
from sklearn.model_selection import cross_val_score

from plot_utils import plot_predicted_vs_actual, plot_precision_recall_curve, plot_roc_curve


def evaluate_classification_model(
        model: BaseEstimator,
        x_test: np.ndarray,
        y_test: np.ndarray,
        display_labels: List[str],
        plot: bool = True
) -> None:
    """
   This function is used to evaluate a classification model on a test set. It calculates and prints out the accuracy,
   precision, recall, and F1-score of the model. If plot is set to True, it also displays a confusion matrix, a
    precision-recall curve, and a ROC curve.

   Parameters:
       model (sklearn.base.BaseEstimator): The machine learning model to evaluate. It should be a trained model.
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
    # Use the model to make predictions on the test set
    y_pred = model.predict(x_test)

    # If the model has a predict_proba method, use it to get the scores. Otherwise, use the decision_function method
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(x_test)
    else:
        y_score = model.decision_function(x_test)

    # Print out the accuracy, precision, recall, and F1-score
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'Precision: {precision_score(y_test, y_pred, average="macro")}')
    print(f'Recall: {recall_score(y_test, y_pred, average="macro")}')
    print(f'F1-score: {f1_score(y_test, y_pred, average="macro")}')

    # Plot a confusion matrix, precision-recall curve, and ROC curve
    if plot:
        ConfusionMatrixDisplay.from_predictions(y_pred, y_test, display_labels=display_labels,
                                                cmap=plt.cm.Blues)
        plot_precision_recall_curve(y_test, y_score, display_labels)
        plot_roc_curve(y_test, y_score, display_labels)


def evaluate_regression_model(
        model: BaseEstimator,
        x_test: np.ndarray,
        y_test: np.ndarray,
        plot: bool = True
) -> None:
    """
    This function is used to evaluate a regression model on a test set. It calculates and prints out the R^2 score,
    Mean Absolute Error, and Root Mean Squared Error of the model. If plot is set to True, it also plots a scatter plot
    of actual vs predicted values.

    Parameters:
        model (sklearn.base.BaseEstimator): The machine learning model to evaluate. It should be a trained model.
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
        plot_predicted_vs_actual(y_test, y_pred)


def perform_cross_validation(
        model: BaseEstimator,
        X: np.ndarray,
        Y: np.ndarray,
        cv: int = 10
) -> np.ndarray:
    """
    This function performs cross validation on a given machine learning model. It returns an array of scores for each
    fold. If the model is a classifier then the accuracy scorer is used, otherwise the root mean squared error is used.

    Parameters:
        model (sklearn.base.BaseEstimator): The machine learning model to be tuned.
        X (np.ndarray): The input data to the model.
        Y (np.ndarray): The target output data for the model.
        cv (int, optional): The number of folds to use for cross validation. Default is 5.

    Returns:
        scores (np.ndarray): An array of scores for each fold.
    """
    # If the model is a classifier, use the accuracy scoring metric, otherwise use the RMSE scoring metric
    if is_classifier(model):
        scores = cross_val_score(model, X, Y, cv=cv, scoring='accuracy', n_jobs=-1)
        print(f'Mean Accuracy: {scores.mean()}')
    else:
        scores = -cross_val_score(model, X, Y, cv=cv, scoring='neg_root_mean_squared_error',  n_jobs=-1)
        print(f'Mean RMSE: {scores.mean()}')
    print(f'Standard Deviation: {scores.std()}')
    return scores


def perform_hypothesis_test(
        model1_scores: np.ndarray,
        model2_scores: np.ndarray,
        significance_level: float = 0.05
) -> None:
    """
    This function performs a paired t-test on the performance of two models. The null hypothesis is that the mean
    performance of model1 is equal to the mean performance of model2. If the p-value is less than the significance
    level, the null hypothesis is rejected and the difference in performance is considered to be statistically
    significant.

    Parameters:
        model1_scores (numpy.ndarray): The performance scores of the first model calculated through cross-validation.
        model2_scores (numpy.ndarray): The performance scores of the second model calculated through cross-validation.
        significance_level (float, optional): The significance level for the hypothesis test. Default is 0.05.

    Returns:
        None
    """
    # Calculate the t-statistic and p-value for a paired t-test
    t_stat, p_value = ttest_rel(model1_scores, model2_scores)
    print(f'T-Statistic: {t_stat}, P-Value: {p_value}')

    # Print out the results of the hypothesis test based on the p-value
    if p_value < significance_level:
        print(f'Assuming a significance level of {significance_level}, the null hypothesis is rejected, and the '
              f'difference in performance is statistically significant')
    else:
        print(f'Assuming a significance level of {significance_level}, the null hypothesis is not rejected, and the '
              f'difference in performance is not statistically significant')


def evaluate_clusters(
        X: np.ndarray,
        Y_pred: np.ndarray,
        Y: np.ndarray = None
) -> None:
    """
    This function evaluates the quality of clustering using several metrics. If true labels are provided, the function
    calculates cluster accuracy by matching each cluster to the most common class in the cluster and then comparing
    this to the true labels. Additionally, the function calculates the Silhouette Score, Davies-Bouldin Index, and
    Calinski-Harabasz Index as measures of cluster quality.

    Parameters:
        X (numpy.ndarray): The samples that have been clustered.
        Y_pred (numpy.ndarray): The predicted cluster labels for each sample.
        Y (numpy.ndarray, optional): The true labels for each sample. If provided, cluster accuracy is calculated.

    Returns:
        None
    """
    # Count the number of clusters
    n_clusters = len(np.unique(Y_pred))

    # Calculate cluster accuracy if true labels are provided
    if Y is not None:
        cluster_labels = -1 * np.ones(n_clusters)

        # Loop through each cluster and find the most common class
        for i in range(n_clusters):
            cluster_samples = Y[Y_pred == i]
            unique_classes, class_counts = np.unique(cluster_samples, return_counts=True)
            cluster_labels[i] = unique_classes[np.argmax(class_counts)]

        # Calculate accuracy of cluster assignments
        accuracy = np.sum(Y_pred == cluster_labels[Y]) / len(Y)
        print(f"Cluster Accuracy: {accuracy}")

    # Calculate Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index as measures of cluster quality
    print('Silhouette Score: ', silhouette_score(X, Y_pred))
    print('Davies-Bouldin Index: ', davies_bouldin_score(X, Y_pred))
    print('Calinski-Harabasz Index: ', calinski_harabasz_score(X, Y_pred))


def optimise_n_kmeans_clusters(
        X: np.ndarray,
        range_n_clusters: range
) -> int:
    """
    This function determines the optimal number of clusters for K-means clustering on a given set of samples by using
    the Silhouette Score. A K-means model is fit for each number of clusters in the given range, and then the Silhouette
    Score is calculated for each model. These scores are plotted on a line graph and displayed. The function then
    returns the number of clusters that resulted in the highest Silhouette Score.

    Parameters:
        X (numpy.ndarray): The set of samples to be clustered.
        range_n_clusters (range): A range of values representing the different numbers of clusters to compute the
                                Silhouette Score for.

    Returns:
        int: The number of clusters that resulted in the highest Silhouette Score.
    """
    # Create a list to store the Silhouette Scores for each number of clusters
    silhouette_scores = []

    # Loop over the given range of numbers of clusters
    for n_clusters in range_n_clusters:
        # Fit a K-means model for the current number of clusters and calculate the Silhouette Score
        km = KMeans(n_clusters=n_clusters, n_init='auto')
        km_Y_pred = km.fit_predict(X)
        # Calculate the Silhouette Score for the current number of clusters and add it to the list
        silhouette_scores.append(silhouette_score(X, km_Y_pred))

    # Plot the Silhouette Scores for each number of clusters
    plt.plot(range_n_clusters, silhouette_scores)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

    # Return the number of clusters that produced the highest Silhouette Score
    return range_n_clusters[silhouette_scores.index(max(silhouette_scores))]


def evaluate_regression_model_for_n_clusters(
        model: BaseEstimator,
        X: np.ndarray,
        Y: np.ndarray,
        Y_pred: np.ndarray,
        n_clusters: int
) -> List[float]:
    """
    This function evaluates a given regression model's performance on each cluster in the dataset separately.
    Cross-validation is used to evaluate the model on each cluster in the dataset and calculates the mean root mean
    squared error (RMSE) for each cluster. These mean RMSEs are then returned in a list.

    Parameters:
        model (sklearn.base.BaseEstimator): The model to be evaluated.
        X (numpy.ndarray): The set of samples that have been clustered.
        Y (numpy.ndarray): The true labels for each sample.
        Y_pred (numpy.ndarray): The predicted cluster labels for each sample.
        n_clusters (int): The number of clusters.

    Returns:
        List[float]: A list of mean RMSEs for each cluster.
    """
    # Create a list to store the mean RMSE for each cluster
    mean_rmse_clusters = []

    # Loop over each cluster
    for i in range(n_clusters):
        # Get the samples and labels for the current cluster
        cluster_X = X[Y_pred == i]
        cluster_Y = Y[Y_pred == i]

        # Cross-validate the model on the current cluster
        cv_scores = -cross_val_score(model, cluster_X, cluster_Y, cv=10, scoring='neg_root_mean_squared_error')

        # Calculate the mean RMSE for the current cluster and add it to the list
        mean_rmse_clusters.append(cv_scores.mean())
        print(f'Mean RMSE for cluster {i}: {cv_scores.mean()}')

    return mean_rmse_clusters
