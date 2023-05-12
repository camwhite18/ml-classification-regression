import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score, f1_score, r2_score, \
    mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC


def evaluate_classification_model(model, x_test, y_test, plot=True):
    y_pred = model.predict(x_test)
    print(f'Accuracy: {model.score(x_test, y_test)}')
    print(f'Precision: {precision_score(y_test, y_pred, average="weighted")}')
    print(f'Recall: {recall_score(y_test, y_pred, average="weighted")}')
    print(f'F1-score: {f1_score(y_test, y_pred, average="weighted")}')
    if plot:
        ConfusionMatrixDisplay.from_predictions(y_pred, y_test, display_labels=['GALAXY', 'QSO', 'STAR'],
                                                cmap=plt.cm.Blues)


def evaluate_regression_model(model, x_test, y_test, plot=True):
    y_pred = model.predict(x_test)
    print(f'R^2 Score: {r2_score(y_test, y_pred)}')
    print(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}')
    print(f'Root Mean Squared Error: {mean_squared_error(y_test, y_pred, squared=False)}')
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


def train_star_support_vector_classifier(x_train, y_train, x_test, y_test):
    model = SVC()
    model.fit(x_train, y_train)
    evaluate_classification_model(model, x_test, y_test)
    return model


def train_gwp_random_forest(x_train, y_train, x_test, y_test):
    model = RandomForestRegressor(n_estimators=50, max_depth=20, min_samples_leaf=4, min_samples_split=2,
                                   bootstrap=True)
    model.fit(x_train, y_train)
    evaluate_regression_model(model, x_test, y_test)
    return model
