from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from IPython.display import display

# Load Iris dataset for demonstration
iris = load_iris()
X = iris.data
y = iris.target

# Define the parameter grid for grid search
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': [0.0001, 0.001, 0.01, 0.1, 1],
}

# Create an SVM classifier
svm = SVC()

# Perform grid search with cross-validation
grid_search = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X, y)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Display a message indicating the completion of the grid search
display("Grid search completed!")