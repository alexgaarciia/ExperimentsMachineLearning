# Import necessary libraries/modules
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier


# Function used to compute metrics
def compute_metrics(classifier, X_test, y_test):
    """This is a function that, given a classifier and the test sets, returns a full metrics report"""
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    return cm, report, accuracy, macro_f1, weighted_f1


# Functions to print information:
def print_info(model_name, cm, report, accuracy, macro_f1, weighted_f1):
    """This function has the main goal of printing information"""
    print("--------------------------------------------------\n")
    print(f"Model Description: {model_name}\n")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-average F1: {macro_f1:.4f}")
    print(f"Weighted-average F1: {weighted_f1:.4f}\n")


def print_info2(model_name, metrics):
    """This function is mainly used to print the average, variance and standard deviation of the accuracy,
    macro_f1 and weighted_f1 of a classifier that is run multiple times"""

    # Compute avg, var, std of accuracy, macro_f1 and weighted f1
    accuracy_avg = np.mean(metrics['accuracies'])
    accuracy_var = np.var(metrics['accuracies'])
    accuracy_std = np.std(metrics['accuracies'])
    macro_f1_avg = np.mean(metrics['macro_f1s'])
    macro_f1_var = np.var(metrics['macro_f1s'])
    macro_f1_std = np.std(metrics['macro_f1s'])
    weighted_f1_avg = np.mean(metrics['weighted_f1s'])
    weighted_f1_var = np.var(metrics['weighted_f1s'])
    weighted_f1_std = np.std(metrics['weighted_f1s'])

    # Print the information
    print(f"Performance summary for {model_name}:")
    print(f"Accuracy: Avg={accuracy_avg}, Var={accuracy_var}, Std={accuracy_std}")
    print(f"Macro F1: Avg={macro_f1_avg}, Var={macro_f1_var}, Std={macro_f1_std}")
    print(f"Weighted F1: Avg={weighted_f1_avg}, Var={weighted_f1_var}, Std={weighted_f1_std}\n")


# Design classifiers
def base_dt(X, X_train, X_test, y_train, y_test):
    # (a) Base-DT
    base_dt_classifier = DecisionTreeClassifier()
    base_dt_classifier.fit(X_train, y_train)

    # Visualize the Decision Tree graphically
    plt.figure(figsize=(12, 8))
    plot_tree(base_dt_classifier, filled=True, feature_names=X.columns, class_names=base_dt_classifier.classes_)
    plt.title("Base-DT Classifier")
    plt.show()

    # Return information needed to compute metrics/show information of training and testing
    return base_dt_classifier, X_test, y_test, "Base-DT"


def top_dt(X, X_train, X_test, y_train, y_test):
    # (b) Top_DT
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    default_dt_classifier = DecisionTreeClassifier()
    grid_search_DT = GridSearchCV(default_dt_classifier, param_grid)
    grid_search_DT.fit(X_train, y_train)

    # Get the best parameters and the best classifier
    top_dt_best_param = grid_search_DT.best_params_
    top_dt_classifier = grid_search_DT.best_estimator_

    # Visualize the Decision Tree graphically
    plt.figure(figsize=(12, 8))
    plot_tree(grid_search_DT.best_estimator_, filled=True, feature_names=X.columns,
              class_names=grid_search_DT.best_estimator_.classes_)
    plt.title("Top-DT Classifier")
    plt.show()

    # Return information needed to compute metrics/show information of training and testing
    return top_dt_classifier, X_test, y_test, 'Top-DT with best parameters' + str(top_dt_best_param)


def base_mlp(X, X_train, X_test, y_train, y_test):
    # (c) Base-MLP
    base_MLP = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic', solver='sgd')
    base_MLP.fit(X_train, y_train)

    # Return information needed to compute metrics/show information of training and testing
    return base_MLP, X_test, y_test, 'Base-MLP'


def top_mlp(X, X_train, X_test, y_train, y_test):
    # (d) Top_MLP
    param_grid_mlp = {
        'activation': ['sigmoid', 'tanh', 'relu'],
        'hidden_layer_sizes': [(100, 100), (10, 10, 10), (30, 50)],
        'solver': ['adam', 'sgd']
    }
    default_MLP = MLPClassifier()
    grid_search_MLP = GridSearchCV(default_MLP, param_grid_mlp)
    grid_search_MLP.fit(X_train, y_train)

    # Get the best parameters and the best classifier
    top_mlp_best_param = grid_search_MLP.best_params_
    top_mlp_classifier = grid_search_MLP.best_estimator_

    # Return information needed to compute metrics/show information of training and testing
    return top_mlp_classifier, X_test, y_test, 'Top-MLP with best parameters' + str(top_mlp_best_param)


# Code to run the models a certain number of iterations
def evaluate_models(X, X_train, X_test, y_train, y_test, num_iterations):
    """The main goal of this function is invoking all the classifiers"""

    # Declare a dictionary that will allow us to store useful information of each classifier
    performance_metrics = {
        'Base-DT': {'accuracies': [], 'macro_f1s': [], 'weighted_f1s': []},
        'Top-DT': {'accuracies': [], 'macro_f1s': [], 'weighted_f1s': []},
        'Base-MLP': {'accuracies': [], 'macro_f1s': [], 'weighted_f1s': []},
        'Top-MLP': {'accuracies': [], 'macro_f1s': [], 'weighted_f1s': []}
    }

    # Train/test the classifiers a specific number of times (num_iterations)
    for _ in range(num_iterations):
        # Call classifiers
        base_dt_class = base_dt(X, X_train, X_test, y_train, y_test)
        top_dt_class = top_dt(X, X_train, X_test, y_train, y_test)
        base_mlp_class = base_mlp(X, X_train, X_test, y_train, y_test)
        top_mlp_class = top_mlp(X, X_train, X_test, y_train, y_test)

        # Compute metrics
        base_dt_metrics = compute_metrics(base_dt_class[0], X_test, y_test)
        top_dt_metrics = compute_metrics(top_dt_class[0], X_test, y_test)
        base_mlp_metrics = compute_metrics(base_mlp_class[0], X_test, y_test)
        top_mlp_metrics = compute_metrics(top_mlp_class[0], X_test, y_test)

        # Print corresponding information
        print_info(base_dt_class[3], base_dt_metrics[0], base_dt_metrics[1], base_dt_metrics[2], base_dt_metrics[3],
                       base_dt_metrics[4])
        print_info(top_dt_class[3], top_dt_metrics[0], top_dt_metrics[1], top_dt_metrics[2], top_dt_metrics[3],
                       top_dt_metrics[4])
        print_info(base_mlp_class[3], base_mlp_metrics[0], base_mlp_metrics[1], base_mlp_metrics[2],
                       base_mlp_metrics[3], base_mlp_metrics[4])
        print_info(top_mlp_class[3], top_mlp_metrics[0], top_mlp_metrics[1], top_mlp_metrics[2], top_mlp_metrics[3],
                       top_mlp_metrics[4])

        # In case the classifiers are trained more than once, we keep the performance of every single iteration
        if num_iterations != 1:
            # Store information
            performance_metrics["Base-DT"]['accuracies'].append(base_dt_metrics[2])
            performance_metrics["Base-DT"]['macro_f1s'].append(top_dt_metrics[3])
            performance_metrics["Base-DT"]['weighted_f1s'].append(top_dt_metrics[4])

            performance_metrics["Top-DT"]['accuracies'].append(base_dt_metrics[2])
            performance_metrics["Top-DT"]['macro_f1s'].append(base_dt_metrics[2])
            performance_metrics["Top-DT"]['weighted_f1s'].append(base_dt_metrics[4])

            performance_metrics["Base-MLP"]['accuracies'].append(base_mlp_metrics[2])
            performance_metrics["Base-MLP"]['macro_f1s'].append(base_mlp_metrics[3])
            performance_metrics["Base-MLP"]['weighted_f1s'].append(base_mlp_metrics[4])

            performance_metrics["Top-MLP"]['accuracies'].append(top_mlp_metrics[2])
            performance_metrics["Top-MLP"]['macro_f1s'].append(top_mlp_metrics[3])
            performance_metrics["Top-MLP"]['weighted_f1s'].append(top_mlp_metrics[4])

    return performance_metrics
