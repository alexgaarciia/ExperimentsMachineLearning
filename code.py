# Import necessary libraries/modules
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier

# Initialize a dictionary to hold all performance metrics
performance_metrics = {
    'Base-DT': {'accuracies': [], 'macro_f1s': [], 'weighted_f1s': []},
    'Top-DT with best parameters': {'accuracies': [], 'macro_f1s': [], 'weighted_f1s': []},
    'Base-MLP': {'accuracies': [], 'macro_f1s': [], 'weighted_f1s': []},
    'Top-MLP with best parameters': {'accuracies': [], 'macro_f1s': [], 'weighted_f1s': []}
}


# Function used to print information
def print_model_metrics(iteration, classifier, X_test, y_test, model_name):
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Append the current scores to the respective lists in the dictionary
    metrics = performance_metrics[model_name]
    metrics['accuracies'].append(accuracy)
    metrics['macro_f1s'].append(macro_f1)
    metrics['weighted_f1s'].append(weighted_f1)

    if iteration == 1:
        print("--------------------------------------------------\n")
        print(f"Model Description: {model_name}\n")
        print("Confusion Matrix:\n", cm)
        print("\nClassification Report:\n", report)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro-average F1: {macro_f1:.4f}")
        print(f"Weighted-average F1: {weighted_f1:.4f}\n")
    else:
        for model_name, metrics in performance_metrics.items():
            accuracy_avg = np.mean(metrics['accuracies'])
            accuracy_var = np.var(metrics['accuracies'])
            accuracy_std = np.std(metrics['accuracies'])
            macro_f1_avg = np.mean(metrics['macro_f1s'])
            macro_f1_var = np.var(metrics['macro_f1s'])
            macro_f1_std = np.std(metrics['macro_f1s'])
            weighted_f1_avg = np.mean(metrics['weighted_f1s'])
            weighted_f1_var = np.var(metrics['weighted_f1s'])
            weighted_f1_std = np.std(metrics['weighted_f1s'])

            print(f"Performance summary for {model_name}:")
            print(f"Accuracy: Avg={accuracy_avg}, Var={accuracy_var}, Std={accuracy_std}")
            print(f"Macro F1: Avg={macro_f1_avg}, Var={macro_f1_var}, Std={macro_f1_std}")
            print(f"Weighted F1: Avg={weighted_f1_avg}, Var={weighted_f1_var}, Std={weighted_f1_std}\n")


# Design classifiers
def models_run(iteration, X, X_train, X_test, y_train, y_test):
    # (a) Base-DT
    base_dt_classifier = DecisionTreeClassifier(random_state=42)
    base_dt_classifier.fit(X_train, y_train)
    print_model_metrics(iteration, base_dt_classifier, X_test, y_test, 'Base-DT')

    # Visualize the Decision Tree graphically
    plt.figure(figsize=(12, 8))
    plot_tree(base_dt_classifier, filled=True, feature_names=X.columns, class_names=base_dt_classifier.classes_)
    plt.title("Base-DT Classifier")
    plt.show()

    # (b) Top_DT
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    default_dt_classifier = DecisionTreeClassifier()
    grid_search_DT = GridSearchCV(default_dt_classifier, param_grid)
    grid_search_DT.fit(X_train, y_train)

    top_dt_best_param = grid_search_DT.best_params_
    top_dt_classifier = grid_search_DT.best_estimator_

    # print("The best parameters of Top-DT classification is:", top_dt_best_param)
    print_model_metrics(iteration, top_dt_classifier, X_test, y_test, 'Top-DT with best parameters')

    # Visualize the Decision Tree graphically
    plt.figure(figsize=(12, 8))
    plot_tree(grid_search_DT.best_estimator_, filled=True, feature_names=X.columns,
              class_names=grid_search_DT.best_estimator_.classes_)
    plt.title("Top-DT Classifier")
    plt.show()

    # (c) Base-MLP
    base_MLP = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic', solver='sgd')
    base_MLP.fit(X_train, y_train)
    print_model_metrics(iteration, base_MLP, X_test, y_test, 'Base-MLP')

    # (d) Top_MLP
    param_grid_mlp = {
        'activation': ['logistic', 'tanh', 'relu'],
        'hidden_layer_sizes': [(100, 100), (10, 10, 10), (30, 50)],
        'solver': ['adam', 'sgd']
    }
    default_MLP = MLPClassifier(max_iter=1500)
    grid_search_MLP = GridSearchCV(default_MLP, param_grid_mlp)
    grid_search_MLP.fit(X_train, y_train)

    top_mlp_best_param = grid_search_MLP.best_params_
    top_mlp_classifier = grid_search_MLP.best_estimator_

    # print("The best parameters of Top-MLP classification is:", top_dt_best_param)
    print_model_metrics(iteration, top_mlp_classifier, X_test, y_test, 'Top-MLP with best parameters')

