# Import necessary libraries/modules
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier


# Function used to print information
def print_model_metrics(classifier, X_test, y_test, model_name):
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')

    print("--------------------------------------------------\n")
    print(f"Model Description: {model_name}\n")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-average F1: {macro_f1:.4f}")
    print(f"Weighted-average F1: {weighted_f1:.4f}\n")


# Design classifiers
def models_run(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    print(X_train, X_test, y_train, y_test)

    # 4. Train and test 4 different classifiers:
    # (a) Base-DT: a Decision Tree with the default parameters. Show the decision tree graphically (for the
    # abalone dataset, you can restrict the tree depth for visualisation purposes)

    # Create a Decision Tree Classifier with default parameters
    base_dt_classifier = DecisionTreeClassifier(random_state=42)
    # Specify max_depth for visualizing the tree

    # Train the classifier on the training data
    base_dt_classifier.fit(X_train, y_train)

    # Test the classifier on the testing data
    accuracy_base_dt = base_dt_classifier.score(X_test, y_test)
    print("The accuracy of Base-DT classification is:", accuracy_base_dt)

    # Visualize the Decision Tree graphically
    plt.figure(figsize=(12, 8))
    plot_tree(base_dt_classifier, filled=True, feature_names=X.columns, class_names=base_dt_classifier.classes_)
    plt.title("Base-DT Classifier")
    plt.show()

    # Simon Starts Writting code like a maniac with no comments what so ever >>> fully autistic mode
    #Top_DT (b)
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

    print("The best parameters of Top-DT classification is:", top_dt_best_param)
    print("The accuracy of Top-DT classification is:", top_dt_classifier.score(X_test, y_test))

    # Visualize the Decision Tree graphically
    plt.figure(figsize=(12, 8))
    plot_tree(top_dt_classifier, filled=True, feature_names=X.columns, class_names=top_dt_classifier.classes_)
    plt.title("Top-DT Classifier")
    plt.show()

    # Base_MLP (c)
    base_MLP = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic', solver='sgd')

    base_MLP.fit(X_train, y_train)

    print("The accuracy of Base_MLP classification is:", base_MLP.score(X_test, y_test))

    # Top_MLP (d)
    param_grid = {
        'activation': ['logistic', 'tanh', 'relu'],
        'hidden_layer_sizes': [(100, 100), (10, 10, 10), (30, 50)],
        'solver': ['adam', 'sgd']
    }

    default_MLP = MLPClassifier(max_iter=1000)
    grid_search_MLP = GridSearchCV(default_MLP, param_grid)

    grid_search_MLP.fit(X_train, y_train)

    top_mlp_best_param = grid_search_MLP.best_params_
    top_mlp_classifier = grid_search_MLP.best_estimator_

    print("The best parameters of Top-MLP classification is:", top_mlp_best_param)
    print("The accuracy of Top-MLP classification is:", top_mlp_classifier.score(X_test, y_test))


# ASK SIMON FOR HELP:
def models_run2(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # (a) Base-DT
    base_dt_classifier = DecisionTreeClassifier(random_state=42)
    base_dt_classifier.fit(X_train, y_train)
    print_model_metrics(base_dt_classifier, X_test, y_test, 'Base-DT')

    # (b) Top_DT
    param_grid_dt = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    grid_search_DT = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5)
    grid_search_DT.fit(X_train, y_train)
    print_model_metrics(grid_search_DT.best_estimator_, X_test, y_test, 'Top-DT with best parameters: ' + str(grid_search_DT.best_params_))

    # (c) Base-MLP
    base_MLP = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic', solver='sgd', random_state=42)
    base_MLP.fit(X_train, y_train)
    print_model_metrics(base_MLP, X_test, y_test, 'Base-MLP')

    # (d) Top_MLP
    param_grid_mlp = {
        'activation': ['logistic', 'tanh', 'relu'],
        'hidden_layer_sizes': [(100, 100), (10, 10, 10), (30, 50)],
        'solver': ['adam', 'sgd']
    }
    grid_search_MLP = GridSearchCV(MLPClassifier(max_iter=1500, random_state=42), param_grid_mlp, cv=5)
    grid_search_MLP.fit(X_train, y_train)
    print_model_metrics(grid_search_MLP.best_estimator_, X_test, y_test, 'Top-MLP with best parameters: ' + str(grid_search_MLP.best_params_))
