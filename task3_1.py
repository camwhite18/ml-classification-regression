import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold


def remove_outliers(X, Y, ignore_cols=None):
    if ignore_cols is None:
        ignore_cols = []
    cols = [i for i in range(X.shape[1]) if i not in ignore_cols]

    # Use the Interquartile Range (IQR) to remove outliers
    Q1 = np.quantile(X[:, cols], 0.25, axis=0)
    Q3 = np.quantile(X[:, cols], 0.75, axis=0)
    IQR = Q3 - Q1

    mask = ~((X[:, cols] < (Q1 - 1.5 * IQR)) | (X[:, cols] > (Q3 + 1.5 * IQR))).any(axis=1)
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


def pearson_correlation_matrix(X, show_coef=True):
    n_features = X.shape[1]
    correlation_matrix = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(n_features):
            correlation_matrix[i, j] = pearson_correlation(X[:, i], X[:, j])

    # Create a heatmap using Matplotlib
    fig, ax = plt.subplots()
    cax = ax.imshow(correlation_matrix, cmap='PuOr', vmin=-1, vmax=1, origin='lower')

    # Add correlation coefficients to each cell
    if show_coef:
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


def standardise_features(X, cols=None):
    # If no columns are specified, standardise all features
    if cols is None:
        cols = range(X.shape[1])

    # Scale the training set using a standard scaler
    standard_scaler = StandardScaler()
    X[:, cols] = standard_scaler.fit_transform(X[:, cols])

    # Check that the mean of each feature transformed is 0 and the standard deviation is 1
    expected_mean = np.zeros(len(cols))
    expected_std = np.ones(len(cols))
    computed_mean = np.mean(X[:, cols], axis=0)
    computed_std = np.std(X[:, cols], axis=0)

    assert np.all(np.round(computed_mean, 6) == expected_mean)
    assert np.all(np.round(computed_std, 6) == expected_std)

    return X


def preprocess_star_dataset(file_name: str = "star_assessment.csv") -> (np.ndarray, np.ndarray):
    star_features = np.genfromtxt(file_name, delimiter=',', skip_header=True, encoding="utf-8", usecols=range(0, 17))
    star_labels = np.genfromtxt(file_name, delimiter=',', skip_header=True, encoding="utf-8", usecols=17, dtype=None)

    # Fill in the missing values
    knn_imputer = KNNImputer()
    star_features = knn_imputer.fit_transform(star_features)

    # Encode the class labels
    label_encoder = LabelEncoder()
    star_labels = label_encoder.fit_transform(star_labels)

    # Remove outliers
    star_features, star_labels = remove_outliers(star_features, star_labels)

    # Remove features with no variance
    star_features, _ = remove_no_variance(star_features)

    # Scale the features
    star_features = standardise_features(star_features)

    # Drop the features that are not relevant to the classification task
    star_features = np.delete(star_features, [0, 1, 2, 8, 9, 10, 15], axis=1)

    return star_features, star_labels


def fill_missing_dates(gwp_categorical):
    for i in range(1, gwp_categorical.shape[0]):
        if not gwp_categorical[i, 0]:
            if gwp_categorical[i - 1, 0] and (gwp_categorical[i - 1, 0] == gwp_categorical[i + 1, 0]):
                gwp_categorical[i, 0] = gwp_categorical[i - 1, 0]
            else:
                previous_day = datetime.datetime.strptime(gwp_categorical[i - 1, 3], '%A').weekday()
                current_day = datetime.datetime.strptime(gwp_categorical[i, 3], '%A').weekday()
                difference_in_days = (current_day - previous_day) % 7

                prev_date = datetime.datetime.strptime(gwp_categorical[i - 1, 0], '%m/%d/%Y')
                current_date = prev_date + datetime.timedelta(days=difference_in_days)
                gwp_categorical[i, 0] = current_date.strftime('%m/%d/%Y')
    return gwp_categorical


def fill_missing_quarters(gwp_categorical):
    for i in range(1, gwp_categorical.shape[0]):
        if not gwp_categorical[i, 1]:
            if gwp_categorical[i - 1, 1] and (gwp_categorical[i - 1, 1] == gwp_categorical[i + 1, 1]):
                gwp_categorical[i, 1] = gwp_categorical[i - 1, 1]
    return gwp_categorical


def fill_missing_days(gwp_categorical):
    for i in range(1, gwp_categorical.shape[0]):
        if not gwp_categorical[i, 3]:
            gwp_categorical[i, 3] = datetime.datetime.strptime(gwp_categorical[i, 0], '%m/%d/%Y').strftime('%A')
    return gwp_categorical


def convert_date_to_cols(X, col):
    for i in range(X.shape[0]):
        X[i, col] = np.datetime64(datetime.datetime.strptime(X[i, col], '%m/%d/%Y'))

    year = X[:, col].astype('datetime64[Y]').astype(int) + 1970
    month = (X[:, col].astype('datetime64[M]').astype(int) % 12) + 1
    day = (X[:, col].astype('datetime64[D]') -
           X[:, col].astype('datetime64[M]')).astype(int) + 1

    X = np.delete(X, col, axis=1)
    # Create a new numpy array with the encoded date features
    X = np.column_stack((year, month, day, X))
    return X


def preprocess_gwp_dataset(file_name: str = "gwp_assessment.csv") -> (np.ndarray, np.ndarray):
    gwp_categorical = np.genfromtxt(file_name, delimiter=',', skip_header=True, encoding="utf-8", dtype=str,
                                    usecols=range(0, 4))
    gwp_numerical = np.genfromtxt(file_name, delimiter=',', skip_header=True, encoding="utf-8", dtype=np.float64,
                                  usecols=range(4, 14))
    gwp_values = np.genfromtxt(file_name, delimiter=',', skip_header=True, encoding="utf-8", dtype=np.float64,
                               usecols=14)

    # Fill in the missing categorical values
    gwp_categorical = fill_missing_dates(gwp_categorical)
    gwp_categorical = fill_missing_quarters(gwp_categorical)
    gwp_categorical = fill_missing_days(gwp_categorical)
    gwp_categorical[:, 2] = np.char.strip(gwp_categorical[:, 2])
    mask = gwp_categorical[:, 2] != ''
    gwp_categorical = gwp_categorical[mask]
    gwp_numerical = gwp_numerical[mask]
    gwp_values = gwp_values[mask]

    # Encode the categorical features
    gwp_categorical = convert_date_to_cols(gwp_categorical, 0)
    gwp_categorical_ohc = gwp_categorical[:, :3].copy()

    for i in range(3, 6):
        ohc = OneHotEncoder(categories='auto', dtype=float, sparse_output=False)
        new_col = ohc.fit_transform(gwp_categorical[:, [i]])
        gwp_categorical_ohc = np.hstack((gwp_categorical_ohc, new_col))

    # Fill in missing numerical values
    gwp_numerical[(gwp_categorical[:, 2] == 'finishing') & (np.isnan(gwp_numerical[:, 3])), 3] = 0
    gwp_features = np.hstack((gwp_categorical_ohc, gwp_numerical))

    knn_imputer = KNNImputer()
    gwp_features = knn_imputer.fit_transform(gwp_features)

    # Encode the numerical features
    gwp_features[:, 16] = np.round(gwp_features[:, 16])
    ohc = OneHotEncoder(categories='auto', dtype=float, sparse_output=False)
    team = ohc.fit_transform(gwp_features[:, [16]])
    gwp_features = np.hstack((gwp_features[:, :16], team, gwp_features[:, 17:]))

    # Remove outliers
    continuous_features = list(range(0, 28)) + list(range(33, 36))
    gwp_features, gwp_values = remove_outliers(gwp_features, gwp_values, ignore_cols=continuous_features)

    # Remove features with no variance
    gwp_features, _ = remove_no_variance(gwp_features)

    # Scale the features
    gwp_features = standardise_features(gwp_features, cols=range(27, 36))
    scaler = StandardScaler()
    gwp_values = scaler.fit_transform(gwp_values.reshape(-1, 1)).flatten()

    # Drop the features that are not relevant to the regression task
    gwp_features = np.delete(gwp_features, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                            11, 12, 13, 14, 29, 30, 31, 32, 33], axis=1)

    return gwp_features, gwp_values
