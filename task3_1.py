import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


def remove_outliers(X, Y):
    # Use the Interquartile Range (IQR) to remove outliers
    Q1 = np.quantile(X, 0.25, axis=0)
    Q3 = np.quantile(X, 0.75, axis=0)
    IQR = Q3 - Q1
    mask = ~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
    X = X[mask]
    Y = Y[mask]
    return X, Y


def remove_no_variance(X):
    # Use a variance threshold to remove any features with zero variance
    threshold = 0
    variance_threshold = VarianceThreshold(threshold=threshold)
    X_transformed = variance_threshold.fit_transform(X)

    # Find the indices of the removed columns
    removed_columns = np.where(variance_threshold.variances_ == threshold)[0]

    return X_transformed, removed_columns


def pearson_correlation(X, Y):
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    num = np.sum((X - x_mean) * (Y - y_mean))
    den = np.sqrt(np.sum((X - x_mean) ** 2) * np.sum((Y - y_mean) ** 2))
    return num / den


def pearson_correlation_matrix(X):
    n_features = X.shape[1]
    correlation_matrix = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(n_features):
            correlation_matrix[i, j] = pearson_correlation(X[:, i], X[:, j])

    # Create a heatmap using Matplotlib
    fig, ax = plt.subplots()
    cax = ax.imshow(correlation_matrix, cmap='PuOr', vmin=-1, vmax=1, origin='lower')

    # Add correlation coefficients to each cell
    for i in range(n_features):
        for j in range(n_features):
            value = round(correlation_matrix[i, j], 2)
            ax.text(j, i, value, ha='center', va='center', color='black' if -0.5 < value < 0.5 else 'white')

    # Set plot title and labels
    ax.set_title("Pearson Correlation Matrix")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Feature Index")
    ax.set_xticks(range(n_features))
    ax.set_yticks(range(n_features))

    # Add colorbar
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
    cbar.set_label('Correlation Coefficient')

    # Show the plot
    plt.show()


def standardise_features(X):
    # Scale the training set using a standard scaler
    standard_scaler = StandardScaler()
    X = standard_scaler.fit_transform(X)

    # Check that the mean of each feature is 0 and the standard deviation is 1
    expected_mean = np.zeros(X.shape[1])
    expected_std = np.ones(X.shape[1])
    computed_mean = np.mean(X, axis=0)
    computed_std = np.std(X, axis=0)

    assert np.all(np.round(computed_mean, 6) == expected_mean)
    assert np.all(np.round(computed_std, 6) == expected_std)

    return X


def preprocess_star_dataset(file_name: str = "star_assessment.csv") -> (np.ndarray, np.ndarray):
    star_features = np.genfromtxt(file_name, delimiter=',', skip_header=True, encoding="utf-8", usecols=range(0, 17))
    star_labels = np.genfromtxt(file_name, delimiter=',', skip_header=True, encoding="utf-8", usecols=17, dtype=None)

    # Immediately drop the features that are not relevant to the classification task
    star_features = np.delete(star_features, [0, 1, 2, 8, 9, 10, 15], axis=1)

    # Fill in the missing values
    knn_imputer = KNNImputer()
    star_features = knn_imputer.fit_transform(star_features)

    # Encode the class labels
    label_encoder = LabelEncoder()
    star_labels = label_encoder.fit_transform(star_labels)

    # Remove outliers
    star_features, star_labels = remove_outliers(star_features, star_labels)

    # Remove features with no variance
    star_features = remove_no_variance(star_features)

    # Scale the features
    star_features = standardise_features(star_features)

    return star_features, star_labels
