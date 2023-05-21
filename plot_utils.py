import math
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.preprocessing import label_binarize


def plot_class_distribution(
        Y: np.ndarray,
) -> None:
    """
    This function is used to plot the class distribution of a dataset on a pie chart.

    Parameters:
        Y (np.ndarray): The labels for the dataset. It should be a 1D numpy array with the same length as the number
                        of samples in X.

    Returns:
        None
    """
    # Check input array
    assert Y.shape[0] > 0, "The number of samples in Y must be greater than 0"

    # Compute the class distribution
    class_distribution = np.unique(Y, return_counts=True)[1]

    # Plot the class distribution
    plt.figure()
    plt.pie(class_distribution, labels=np.unique(Y), autopct='%1.1f%%')
    plt.title('Class Distribution')
    plt.show()


def plot_feature_boxplot(
        X: np.ndarray,
        X_labels: List[str],
        ignore_cols: List[int] = None
) -> None:
    """
    This function is used to plot a boxplot for each feature in a dataset.

    Parameters:
        X (np.ndarray): The dataset to plot. It should be a 2D numpy array with the shape (n_samples, n_features).
        X_labels (List[str]): The labels to display on the x-axis. It should be a list of strings, where the strings
                            are the names of the features in the dataset.
        ignore_cols (List[int], optional): The indices of the columns to ignore. If None, then no columns are ignored.

    Returns:
        None
    """
    # Check input array
    assert X.shape[0] > 0, "The number of samples in X must be greater than 0"
    assert X.shape[1] > 0, "The number of features in X must be greater than 0"
    assert len(X_labels) == X.shape[1], "The number of labels in X_labels must match the number of features in X"

    # Remove the columns that are to be ignored
    if ignore_cols is None:
        ignore_cols = []
    cols = [i for i in range(X.shape[1]) if i not in ignore_cols]

    # Determine the number of rows and columns for the subplots
    num_rows = math.ceil(len(cols) / 4)
    num_cols = 4

    # Plot subplots for each feature
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows))
    ax, i = ax.flatten(), 0
    for i, col in enumerate(cols):
        ax[i].boxplot(X[:, col])
        ax[i].set_title(X_labels[col])

    # Remove the unused subplots
    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(ax[j])
    plt.show()


def plot_feature_scores(
        scores: np.ndarray,
        feature_labels: np.ndarray,
        metric: str
) -> None:
    """
    This function plots the feature scores on a bar chart based on a given metric.

    Parameters:
        scores (np.ndarray): The scores for each feature.
        feature_labels (np.ndarray): The labels for each feature.
        metric (str): The metric used to score the features.

    Returns:
        None
    """
    # Check input arrays
    assert scores.shape[0] > 0, "The number of scores must be greater than 0"
    assert len(feature_labels) == scores.shape[0], "The number of labels must match the number of scores"

    # Sort feature names based on the scores
    sorted_scores = np.argsort(scores)[::-1]
    sorted_feature_labels = [feature_labels[i] for i in sorted_scores]

    # Plot the sorted scores
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(sorted_feature_labels)), scores[sorted_scores])
    plt.xticks(ticks=range(len(sorted_feature_labels)), labels=sorted_feature_labels, rotation=90)
    plt.xlabel('Feature')
    plt.ylabel(f'{metric} score')
    plt.title(f'{metric} scores for each feature')
    plt.grid(True)

    # Display the plot
    plt.show()


def pearson_correlation(
        X: np.ndarray,
        Y: np.ndarray
) -> float:
    """
    This function computes the Pearson correlation coefficient between two lists of numbers.

    Parameters:
        X (np.ndarray): The first list of numbers.
        Y (np.ndarray): The second list of numbers.

    Returns:
        float: The Pearson correlation coefficient between X and Y.
    """
    # Check input arrays
    assert X.shape[0] > 0, "The number of samples in X must be greater than 0"
    assert Y.shape[0] > 0, "The number of samples in Y must be greater than 0"
    assert X.shape[0] == Y.shape[0], "The number of samples in X and Y must be equal"

    # Calculate the mean of X and Y
    x_mean = np.mean(X)
    y_mean = np.mean(Y)

    # Calculate the numerator and denominator of the Pearson correlation coefficient
    numerator = np.sum((X - x_mean) * (Y - y_mean))
    denominator = np.sqrt(np.sum((X - x_mean) ** 2) * np.sum((Y - y_mean) ** 2))

    # Return the Pearson correlation coefficient
    return numerator / denominator


def plot_feature_correlations(
        X: np.ndarray,
        X_labels: List[str],
        show_coef: bool = True
) -> None:
    """
    This function plots a Pearson correlation matrix of features in the form of a heatmap.

    Parameters:
        X (np.ndarray): The samples to plot the correlation matrix for.
        X_labels (list of str): The labels for each feature.
        show_coef (bool, optional): Whether to show the correlation coefficient in each cell. Defaults to True.

    Returns:
        None
    """
    # Check input array
    assert X.shape[0] > 0, "The number of samples in X must be greater than 0"
    assert X.shape[1] > 0, "The number of features in X must be greater than 0"
    assert len(X_labels) == X.shape[1], "The number of labels in X_labels must match the number of features in X"

    # Create a matrix filled with zeros
    num_features = X.shape[1]
    correlation_matrix = np.zeros((num_features, num_features))

    # Calculate the Pearson correlation coefficient for each pair of features and store it in the matrix
    for i in range(num_features):
        for j in range(num_features):
            correlation_matrix[i, j] = pearson_correlation(X[:, i], X[:, j])

    # Create a heatmap from the matrix using Matplotlib
    fig, ax = plt.subplots()
    cax = ax.imshow(correlation_matrix, cmap='PuOr', vmin=-1, vmax=1, origin='lower')

    # Add correlation coefficients to each cell
    if show_coef:
        for i in range(num_features):
            for j in range(num_features):
                value = round(correlation_matrix[i, j], 2)
                ax.text(j, i, value, ha='center', va='center', color='black' if -0.5 < value < 0.5 else 'white')

    # Set plot title and labels
    ax.set_title("Pearson Correlation Matrix")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Feature Index")
    ax.set_xticks(range(num_features))
    ax.set_yticks(range(num_features))
    ax.set_xticklabels(X_labels, rotation=90)
    ax.set_yticklabels(X_labels)

    # Add colour bar
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
    cbar.set_label('Correlation Coefficient')

    # Show the plot
    plt.show()


def plot_precision_recall_curve(
        y_test: np.ndarray,
        y_score: np.ndarray,
        class_labels: List[str]
) -> None:
    """
    This function is used to plot a precision-recall curve for a classification problem with multiple classes.

    Parameters:
        y_test (np.ndarray): The true labels for the test set. It should be a 1D numpy array with the same length as
                            the number of samples in x_test.
        y_score (np.ndarray): The scores for the test set. It should be a 2D numpy array, where the number of columns
                            is equal to the number of classes, and each column represents the scores for that class.
        class_labels (List[str]): The labels to display on the confusion matrix. It should be a list of strings, where
                            the strings are the names of the classes in the dataset.

    Returns:
        None
    """
    # Check input arrays
    assert y_test.shape[0] == y_score.shape[0], "The number of samples between y_test and y_score do not match"
    assert y_score.shape[1] == len(class_labels), "The number of classes between y_score and class_labels do not match"

    # Binarise the labels
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

    # Plot the precision-recall curve for each class
    plt.figure()
    precision, recall = dict(), dict()
    for i in range(len(class_labels)):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        plt.plot(recall[i], precision[i], lw=2, label=f'{class_labels[i]}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='best')
    plt.title('Precision-Recall Curve')
    plt.show()


def plot_roc_curve(
        y_test: np.ndarray,
        y_score: np.ndarray,
        class_labels: List[str]
) -> None:
    """
    This function is used to plot a ROC curve for a classification problem with multiple classes.

    Parameters:
        y_test (np.ndarray): The true labels for the test set. It should be a 1D numpy array with the same length as
                            the number of samples in x_test.
        y_score (np.ndarray): The scores for the test set. It should be a 1D numpy array with the same length as
                            the number of samples in x_test.
        class_labels (List[str]): The labels to display on the confusion matrix. It should be a list of strings, where
                            the strings are the names of the classes in the dataset.

    Returns:
        None
    """
    # Binarise the labels
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

    # Plot the ROC curve for each class
    plt.figure()
    false_positive_rate, true_positive_rate = dict(), dict()
    for i in range(len(class_labels)):
        false_positive_rate[i], true_positive_rate[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        plt.plot(false_positive_rate[i], true_positive_rate[i], lw=2, label=f'{class_labels[i]}')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.title('ROC Curve')
    plt.show()


def plot_predicted_vs_actual(
        y_test: np.ndarray,
        y_pred: np.ndarray
) -> None:
    """
    This function is used to plot a predicted vs actual values scatter graph for a regression problem.

    Parameters:
        y_test (np.ndarray): The true labels for the test set. It should be a 1D numpy array with the same length as
                            the number of samples in x_test.
        y_pred (np.ndarray): The scores for the test set. It should be a 1D numpy array with the same length as
                            the number of samples in x_test.

    Returns:
        None
    """
    plt.figure()
    max_val = max(np.max(y_test), np.max(y_pred))
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.xlim([0, max_val])
    plt.ylim([0, max_val])
    plt.title('Actual vs Predicted Values')
    plt.plot([0, max_val], [0, max_val], 'g--')
    plt.show()


def plot_clusters(
        x: np.ndarray,
        y_pred: np.ndarray,
        y: np.ndarray = None
) -> None:
    """
    This function is used to plot the clusters identified by a clustering algorithm alongside the actual labels.

    Parameters:
        x (numpy.ndarray): The input data to be plotted.
        y_pred (numpy.ndarray): The predicted labels from the clustering algorithm.
        y (numpy.ndarray, optional): The actual labels of the data. If provided, a second plot showing the actual
                                    labels is displayed.

    Returns:
        None
    """
    # Apply PCA and transform the data to 2 dimensions
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x)

    # Create the figure and axes
    if y is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax = [ax]
    else:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Scatter plot for the clusters created by K-means
    ax[0].scatter(x_pca[:, 0], x_pca[:, 1], cmap=plt.cm.viridis, c=y_pred)
    ax[0].set_title('K-means Clusters')

    if y is not None:
        # Scatter plot for the actual labels if y is provided
        ax[1].scatter(x_pca[:, 0], x_pca[:, 1], cmap=plt.cm.inferno, c=y)
        ax[1].set_title('Actual Labels')

    plt.show()
