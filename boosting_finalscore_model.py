import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
df = pd.read_csv('boosting_finalscore.csv')
X = df.drop('final_score', axis=1)
y = df['final_score']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Predict + evaluate
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f'RMSE: {rmse:.2f}')
print('Predictions:', preds)
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.05, 0.1],
    'max_depth': [2, 3]
}

# Grid search
grid = GridSearchCV(GradientBoostingRegressor(), param_grid, cv=3, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

# Best model
best_model = grid.best_estimator_
best_preds = best_model.predict(X_test)
best_rmse = np.sqrt(mean_squared_error(y_test, best_preds))

print(f'Best RMSE: {best_rmse:.2f}')
print('Best Params:', grid.best_params_)


