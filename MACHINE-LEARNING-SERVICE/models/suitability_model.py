import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Sample data for training
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0, 1, 0, 1])

# Train a simple model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model
joblib.dump(model, 'suitability_model.pkl')