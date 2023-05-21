import numpy as np
from sklearn.base import is_classifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC


def create_train_test_split(
        X: np.ndarray,
        Y: np.ndarray,
        stratify: bool = False
) -> (np.ndarray, np.ndarray):
    """
    This function splits the input data X and target output Y into training and testing sets. A split of 80% training
    and 20% testing is used with a random state set for reproducibility.

    Parameters:
        X (np.ndarray): The input data to the model.
        Y (np.ndarray): The target output data for the model.
        stratify (bool, optional): Whether to use stratified sampling. The default is False.

    Returns:
        x_train (np.ndarray): The training input data.
        x_test (np.ndarray): The testing input data.
        y_train (np.ndarray): The training target output data.
        y_test (np.ndarray): The testing target output data.
    """
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=1,
        stratify=Y if stratify else None
    )

    return x_train, x_test, y_train, y_test


def tune_hyperparameters(
        model: object,
        param_grid: dict,
        X: np.ndarray,
        Y: np.ndarray,
) -> dict:
    """
    This function uses GridSearchCV to tune the parameters of a given machine learning model by computing a scoring
    metric for each combination of parameters in the param_grid. The parameters with the best score are returned.

    Parameters:
        model (object): The machine learning model to be tuned.
        param_grid (dict): The parameters to be tuned for the model. It is a dictionary where the keys are the
                       parameters and the values are the range of values for each parameter.
        X (np.ndarray): The input data to the model.
        Y (np.ndarray): The target output data for the model.

    Returns:
        best_params (dict): The best parameters found by GridSearchCV.
    """
    # Get the scoring metric for the model
    if is_classifier(model):
        scoring = 'accuracy'
    else:
        scoring = 'neg_root_mean_squared_error'

    # Create a GridSearchCV object with 5-fold cross validation and fit it to the data
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X, Y)

    # Print out the best parameters
    print(f'Best Parameters:')
    for k, v in grid_search.best_params_.items():
        print(f'{k} = {v}')

    return grid_search.best_params_


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
    return SVC(
        kernel='rbf',
        C=1000,
        gamma=0.1
    )


def create_tuned_gwp_rf() -> RandomForestRegressor:
    """
    This function creates a Random Forest regressor with its hyperparameters tuned on the garment worker productivity
    dataset. The hyperparameters are tuned using a grid search with 5-fold cross validation. The hyperparameters are
    tuned as follows:
        - n_estimators: 200
        - max_depth: 10
        - min_samples_leaf: 4
        - min_samples_split: 2
        - bootstrap: True

    Returns:
        RandomForestRegressor: The tuned Random Forest regressor.
    """
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=4,
        min_samples_split=2,
        bootstrap=True
    )


def create_tuned_star_gb() -> GradientBoostingClassifier:
    """
    This function creates a Gradient Boosting classifier with its hyperparameters tuned on the stellar classification
    dataset. The hyperparameters are tuned using a grid search with 5-fold cross validation. The hyperparameters are
    tuned as follows:
        - learning_rate: 0.1
        - max_depth: 5
        - min_samples_leaf: 4
        - min_samples_split: 5
        - n_estimators: 200

    Returns:
        GradientBoostingClassifier: The tuned Gradient Boosting classifier.
    """
    return GradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=5,
        min_samples_leaf=4,
        min_samples_split=5,
        n_estimators=200
    )


def create_tuned_gwp_mlp() -> MLPRegressor:
    """
    This function creates a Multi-Layer Perceptron regressor with its hyperparameters tuned on the garment worker
    productivity dataset. The hyperparameters are tuned using a grid search with 5-fold cross validation. The
    hyperparameters are tuned as follows:
        - activation: tanh
        - solver: adam
        - alpha: 0.05
        - hidden_layer_sizes: (50, 50, 50)
        - learning_rate: adaptive

    Returns:
        MLPRegressor: The tuned Multi-Layer Perceptron regressor.
    """
    return MLPRegressor(
        activation='tanh',
        solver='adam',
        alpha=0.05,
        hidden_layer_sizes=(50, 50, 50),
        learning_rate='adaptive'
    )
