import datetime

import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold


def remove_iqr_outliers(
        X: np.ndarray,
        Y: np.ndarray,
        ignore_cols: np.ndarray = None
) -> (np.ndarray, np.ndarray):
    """
    This function removes outliers from the dataset using the interquartile range (IQR). It does this by calculating
    the IQR for each feature and then applying a mask to remove any data points that lie below Q1 - 1.5 * IQR or above
    Q3 + 1.5 * IQR.

    Parameters:
        X (np.ndarray): The features of the dataset.
        Y (np.ndarray): The labels of the dataset.
        ignore_cols (np.ndarray): The columns to ignore when removing outliers.

    Returns:
        X (np.ndarray): The features of the dataset with outliers removed.
        Y (np.ndarray): The labels of the dataset with outliers removed.
    """
    if ignore_cols is None:
        ignore_cols = []
    cols = [i for i in range(X.shape[1]) if i not in ignore_cols]

    # Calculate the interquartile range (IQR) for each feature
    Q1 = np.quantile(X[:, cols], 0.25, axis=0)
    Q3 = np.quantile(X[:, cols], 0.75, axis=0)
    IQR = Q3 - Q1

    # Apply a mask to remove any data points that lie below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR
    mask = ~((X[:, cols] < (Q1 - 1.5 * IQR)) | (X[:, cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    X = X[mask]
    Y = Y[mask]
    return X, Y


def remove_no_variance(
        X: np.ndarray,
        X_labels: np.ndarray = None
) -> (np.ndarray, np.ndarray):
    """
    This functions removes any features with zero variance from the dataset. This is done by using the variance
    threshold from Scikit-Learn and setting the threshold to zero.

    Parameters:
        X (np.ndarray): The features of the dataset.
        X_labels (np.ndarray): The labels of the dataset.

    Returns:
    X (np.ndarray): The feature matrix with zero variance features removed.
    """
    # Use a variance threshold to remove any features with zero variance
    variance_threshold = VarianceThreshold(threshold=0)
    X = variance_threshold.fit_transform(X)

    if X_labels is not None:
        # Get the mask of the selected features
        mask = variance_threshold.get_support()

        # Remove the labels that correspond to the removed columns
        X_labels = np.array(X_labels)
        X_labels = X_labels[mask]

    return X, X_labels


def preprocess_star_dataset(
        file_name: str = "star_assessment.csv"
) -> (np.ndarray, np.ndarray):
    """
    This function performs the preprocessing steps for the Star dataset with the intent of being used in other notebooks
    to enable code reusability. The steps performed are:
        1. Load the data from the CSV file
        2. Fill in the missing values using KNN imputation
        3. Encode the class labels
        4. Remove outliers using the interquartile range (IQR)
        5. Remove features with no variance
        6. Scale the features using standardization
        7. Drop the features that were not selected in the feature selection process

    Parameters:
        file_name (str): The name of the CSV file containing the Star dataset.

    Returns:
        star_features (np.ndarray): The preprocessed Star features.
        star_labels (np.ndarray): The preprocessed Star labels.
    """
    # Load the data from the CSV file
    star_features = np.genfromtxt(file_name, delimiter=',', skip_header=True, encoding="utf-8", usecols=range(0, 17))
    star_labels = np.genfromtxt(file_name, delimiter=',', skip_header=True, encoding="utf-8", usecols=17, dtype=None)

    # Fill in the missing values
    knn_imputer = KNNImputer()
    star_features = knn_imputer.fit_transform(star_features)

    # Encode the class labels
    label_encoder = LabelEncoder()
    star_labels = label_encoder.fit_transform(star_labels)

    # Remove outliers
    star_features, star_labels = remove_iqr_outliers(star_features, star_labels)

    # Remove features with no variance
    star_features, _ = remove_no_variance(star_features)

    # Scale the features
    std_scaler = StandardScaler()
    star_features = std_scaler.fit_transform(star_features)

    # Drop the features that were not selected in the feature selection process
    star_features = np.delete(star_features, [0, 1, 2, 8, 9, 10, 15], axis=1)

    return star_features, star_labels


def fill_gwp_missing_dates(
        gwp_categorical: np.ndarray
) -> np.ndarray:
    """
    This function fills in the missing dates in the GWP dataset. It does this by looking at the dates in the previous
    and next rows and if they are the same, it fills in the missing date with the same date. If the dates are different,
    it fills in the missing date using the previous date and the difference in days between the previous and next date.

    Parameters:
        gwp_categorical (np.ndarray): The GWP categorical features with missing dates.

    Returns:
        gwp_categorical (np.ndarray): The GWP categorical features with missing dates filled in.
    """
    # Loop through the rows in the GWP categorical features
    for i in range(1, gwp_categorical.shape[0]):
        # If the date is missing
        if not gwp_categorical[i, 0]:
            # If the previous and next dates are the same, fill in the missing date with the same date
            if gwp_categorical[i - 1, 0] and (gwp_categorical[i - 1, 0] == gwp_categorical[i + 1, 0]):
                gwp_categorical[i, 0] = gwp_categorical[i - 1, 0]
            # Otherwise fill in the missing date using the previous date and the difference in days
            else:
                # Compute the difference in days between the previous and next date
                previous_day = datetime.datetime.strptime(gwp_categorical[i - 1, 3], '%A').weekday()
                current_day = datetime.datetime.strptime(gwp_categorical[i, 3], '%A').weekday()
                difference_in_days = (current_day - previous_day) % 7

                # Compute the new date and add it to the numpy array
                prev_date = datetime.datetime.strptime(gwp_categorical[i - 1, 0], '%m/%d/%Y')
                current_date = prev_date + datetime.timedelta(days=difference_in_days)
                gwp_categorical[i, 0] = current_date.strftime('%m/%d/%Y')
    return gwp_categorical


def fill_gwp_missing_quarters(
        gwp_categorical: np.ndarray
) -> np.ndarray:
    """
    This function fills in the missing quarters in the GWP dataset. It does this by looking at the quarters in the
    previous and next rows and if they are the same, it fills in the missing quarter with the same quarter. Since this
    covers all cases of their being missing quarters, this is enough to fill in all the missing values in the column.

    Parameters:
        gwp_categorical (np.ndarray): The GWP categorical features with missing quarters.

    Returns:
        gwp_categorical (np.ndarray): The GWP categorical features with missing quarters filled in.
    """
    # Loop through the rows in the GWP categorical features
    for i in range(1, gwp_categorical.shape[0]):
        # If the quarter is missing
        if not gwp_categorical[i, 1]:
            # If the previous and next quarters are the same, fill in the missing quarter with the same quarter
            if gwp_categorical[i - 1, 1] and (gwp_categorical[i - 1, 1] == gwp_categorical[i + 1, 1]):
                gwp_categorical[i, 1] = gwp_categorical[i - 1, 1]
    return gwp_categorical


def fill_gwp_missing_days(
        gwp_categorical: np.ndarray
) -> np.ndarray:
    """
    This function fills in the missing days in the GWP dataset. It does this by looking at the date in the same row and
    calculates the day from that date. It then fills in the missing day with the calculated day.

    Parameters:
        gwp_categorical (np.ndarray): The GWP categorical features with missing days.

    Returns:
        gwp_categorical (np.ndarray): The GWP categorical features with missing days filled in.
    """
    # Loop through the rows in the GWP categorical features
    for i in range(1, gwp_categorical.shape[0]):
        # If the day is missing
        if not gwp_categorical[i, 3]:
            # Compute the day from the date and add it to the numpy array
            gwp_categorical[i, 3] = datetime.datetime.strptime(gwp_categorical[i, 0], '%m/%d/%Y').strftime('%A')
    return gwp_categorical


def handle_gwp_department(
        gwp_categorical: np.ndarray,
        wip_column: np.ndarray
) -> np.ndarray:
    """
    This function handles the GWP dataset department feature. It first removes the trailing whitespace from any rows
    that have it and fixes the spelling of 'sewing'. It then uses the wip column to infer the department for any rows
    where the department is missing. If the wip column is empty, the department is set to finishing, otherwise it is
    set to sewing.

    Parameters:
        gwp_categorical (np.ndarray): The GWP categorical features with initial departments.
        wip_column (np.array): The wip column from the GWP dataset.

    Returns:
        gwp_categorical (np.ndarray): The GWP categorical features with missing departments filled in and mistakes
                                    fixed.
    """
    # Remove the trailing space from the `finishing` values
    gwp_categorical[:, 2] = np.char.strip(gwp_categorical[:, 2])

    # Fix the spelling of `sewing`
    gwp_categorical[:, 2] = np.where(gwp_categorical[:, 2] == 'sweing', 'sewing', gwp_categorical[:, 2])

    # Loop through the rows in the GWP categorical features
    for i in range(gwp_categorical.shape[0]):
        # If the department is missing
        if not gwp_categorical[i, 2]:
            # If the wip column is empty, set the department to finishing, otherwise set it to sewing
            if not wip_column[i]:
                gwp_categorical[i, 2] = 'finishing'
            else:
                gwp_categorical[i, 2] = 'sewing'
    return gwp_categorical


def convert_date_to_cols(
        X: np.ndarray,
        col: int,
        X_labels: np.ndarray = None
) -> (np.ndarray, np.ndarray):
    """
    This function converts a date column in a numpy array to three columns, one for the year, one for the month, and one
    for the day. It then returns the modified numpy array and the new labels for the columns.

    Parameters:
        X (np.ndarray): The numpy array containing the date column.
        col (int): The index of the date column.
        X_labels (np.ndarray): The labels for the columns in the numpy array.

    Returns:
        X (np.ndarray): The numpy array with the date column converted to three columns.
        X_labels (np.ndarray): The labels for the columns in the numpy array.
    """
    # Loop through the rows in the numpy array and convert each to a datetime object
    for i in range(X.shape[0]):
        X[i, col] = np.datetime64(datetime.datetime.strptime(X[i, col], '%m/%d/%Y'))

    # Extract the year, month, and day from the datetime objects
    year = X[:, col].astype('datetime64[Y]').astype(int) + 1970
    month = (X[:, col].astype('datetime64[M]').astype(int) % 12) + 1
    day = (X[:, col].astype('datetime64[D]') - X[:, col].astype('datetime64[M]')).astype(int) + 1

    # Modify the feature matrix to remove the date column and add the year, month, and day columns
    X = np.delete(X, col, axis=1)
    X = np.column_stack((year, month, day, X))

    # Modify the feature labels to remove the date label and add the year, month, and day labels
    if X_labels is not None:
        X_labels = np.delete(X_labels, col)
        X_labels = np.insert(X_labels, 0, ['year', 'month', 'day'])
    return X, X_labels


def preprocess_gwp_dataset(
        file_name: str = "gwp_assessment.csv"
) -> (np.ndarray, np.ndarray):
    """
    This function performs the preprocessing steps for the GWP dataset with the intent of being used in other notebooks
    to enable code reusability. The steps performed are:
        1. Load the data from the CSV file
        2. Fill in the missing categorical values
        4. Convert the date feature to year, month, and day features
        5. One-hot encode the categorical features
        6. Fill in the missing numerical values using KNN imputation
        7. One-hot encode the team feature
        8. Remove outliers using the interquartile range (IQR)
        9. Remove features with no variance
        10. Scale the features using standardization
        11. Drop the features that were not selected in the feature selection process

    Parameters:
        file_name (str): The name of the CSV file containing the GWP dataset.

    Returns:
        gwp_features (np.ndarray): The preprocessed GWP features.
        gwp_values (np.ndarray): The preprocessed GWP values.
    """
    # Load the data from the CSV file
    gwp_categorical = np.genfromtxt(file_name, delimiter=',', skip_header=True, encoding="utf-8", dtype=str,
                                    usecols=range(0, 4))
    gwp_numerical = np.genfromtxt(file_name, delimiter=',', skip_header=True, encoding="utf-8", dtype=np.float64,
                                  usecols=range(4, 14))
    gwp_values = np.genfromtxt(file_name, delimiter=',', skip_header=True, encoding="utf-8", dtype=np.float64,
                               usecols=14)

    # Fill in the missing categorical values
    gwp_categorical = fill_gwp_missing_days(fill_gwp_missing_quarters(fill_gwp_missing_dates(gwp_categorical)))

    # Handle the department feature
    gwp_categorical = handle_gwp_department(gwp_categorical, gwp_numerical[:, 3])

    # Convert the date to year, month, and day columns
    gwp_categorical, _ = convert_date_to_cols(gwp_categorical, 0)

    # One-hot encode the categorical features
    ohc = OneHotEncoder(categories='auto', dtype=float, sparse_output=False)
    gwp_categorical_ohc = np.hstack((gwp_categorical[:, [0, 1, 2]], ohc.fit_transform(gwp_categorical[:, [3, 4, 5]])))

    # Fill in missing values in the `wip` column
    gwp_numerical[(gwp_categorical[:, 4] == 'finishing') & (np.isnan(gwp_numerical[:, 3])), 3] = 0

    # Combine the categorical and numerical features into a single feature matrix
    gwp_features = np.hstack((gwp_categorical_ohc, gwp_numerical)).astype(float)

    # Impute the missing values
    knn_imputer = KNNImputer()
    gwp_features = knn_imputer.fit_transform(gwp_features)

    # One-hot encode the team feature
    gwp_features[:, 16] = np.round(gwp_features[:, 16])
    ohc = OneHotEncoder(categories='auto', dtype=float, sparse_output=False)
    gwp_features = np.hstack((gwp_features[:, :16], ohc.fit_transform(gwp_features[:, [16]]), gwp_features[:, 17:]))

    # Remove outliers
    gwp_features, gwp_values = remove_iqr_outliers(
        gwp_features, gwp_values, ignore_cols=list(range(0, 28)) + list(range(33, 36))
    )

    # Remove features with no variance
    gwp_features, _ = remove_no_variance(gwp_features)

    # Scale the features
    std_scaler = StandardScaler()
    gwp_features = std_scaler.fit_transform(gwp_features)

    # Drop the features that are not relevant to the regression task
    gwp_features = gwp_features[:, list(range(15, 27)) + [0, 27, 28, 31, 33, 34]]

    return gwp_features, gwp_values
