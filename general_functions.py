import pandas
from numpy import ndarray
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


def train_model(
        model: [HistGradientBoostingClassifier, RandomForestClassifier],
        descriptors: pandas.DataFrame, targets: pandas.DataFrame, test_size: float,
        stratify: bool = True, random_state: int = 42, print_results=True
) -> [BaseEstimator, float, float, ndarray]:

    # Make a copy of the model.
    model = clone(model)

    # Verify the training and test set ratios.
    if test_size > 0.90:
        raise RuntimeError(f"The maximum training-test set ratio can be 1:9.")

    # Set the stratification option.
    stratify = None if stratify is False else targets

    # Create the training and test sets.
    descriptors_train, descriptors_test, targets_train, targets_test = train_test_split(
        descriptors, targets, test_size=test_size, stratify=stratify, random_state=random_state
    )

    # Train the model.
    model.fit(descriptors_train, targets_train)

    # Predict the test set.
    predictions = model.predict(descriptors_test)

    # Calculate the statistics for the test set.
    accuracy = accuracy_score(targets_test, predictions)
    kappa = cohen_kappa_score(targets_test, predictions)
    confusion_table = confusion_matrix(targets_test, predictions)

    if print_results is True:
        print(f'Accuracy: {accuracy}, Kappa: {kappa}')
        print(f'Confusion matrix: {confusion_table[0][0]} {confusion_table[0][1]}')
        print(f'                  {confusion_table[1][0]} {confusion_table[1][1]}')

    # Return the model and the statistics measures.
    return model, accuracy, kappa, confusion_table


def test_model(model: [HistGradientBoostingClassifier, RandomForestClassifier],
               descriptors: pandas.DataFrame, targets: pandas.DataFrame,
               print_results: bool = True
) -> [float, float, ndarray, ndarray, ndarray]:

    # Make predictions.
    predictions = model.predict(descriptors)
    probabilities = model.predict_proba(descriptors)
    # probabilities = model._predict_proba_lr(descriptors)

    # Calculate the statistics for the external test set.
    accuracy = accuracy_score(targets, predictions)
    kappa = cohen_kappa_score(targets, predictions)
    confusion_table = confusion_matrix(targets, predictions)

    if print_results is True:
        print(f'Accuracy: {accuracy}, Kappa: {kappa}')
        print(f'Confusion matrix: {confusion_table[0][0]} {confusion_table[0][1]}')
        print(f'                  {confusion_table[1][0]} {confusion_table[1][1]}')

    # Return the statistics measures.
    return accuracy, kappa, confusion_table, predictions, probabilities


def target_shuffling(
        model: [HistGradientBoostingClassifier, RandomForestClassifier],
        descriptors: pandas.DataFrame, targets: pandas.DataFrame, test_size: float,
        stratify: bool = True, random_state: int = 42, print_results=True
) -> [BaseEstimator, float, float, ndarray]:

    # Shuffle the targets.
    targets = targets.sample(frac=1, random_state=42).reset_index(drop=True)

    # Make a copy of the model.
    model = clone(model)

    # Verify the training and test set ratios.
    if test_size > 0.90:
        raise RuntimeError(f"The maximum training-test set ratio can be 1:9.")

    # Set the stratification option.
    stratify = None if stratify is False else targets

    # Create the training and test sets.
    descriptors_train, descriptors_test, targets_train, targets_test = train_test_split(
        descriptors, targets, test_size=test_size, stratify=stratify, random_state=random_state
    )

    # Train the model.
    model.fit(descriptors_train, targets_train)

    # Predict the test set.
    predictions = model.predict(descriptors_test)

    # Calculate the statistics for the test set.
    accuracy = accuracy_score(targets_test, predictions)
    kappa = cohen_kappa_score(targets_test, predictions)
    confusion_table = confusion_matrix(targets_test, predictions)

    if print_results is True:
        print(f'Accuracy: {accuracy}, Kappa: {kappa}')
        print(f'Confusion matrix: {confusion_table[0][0]} {confusion_table[0][1]}')
        print(f'                  {confusion_table[1][0]} {confusion_table[1][1]}')

    # Return the model and the statistics measures.
    return model, accuracy, kappa, confusion_table
