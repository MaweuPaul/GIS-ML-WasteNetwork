import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from utils.database import engine

def load_data():
    """
    Load data from the database for training.
    """
    query = """
    SELECT * FROM "AreaOfInterest" WHERE geom IS NOT NULL
    """
    df = pd.read_sql(query, engine)
    # Example target variable: 'suitable' (adjust based on your actual target)
    if 'suitable' not in df.columns:
        raise ValueError("Target variable 'suitable' not found in data.")
    X = df.drop(['id', 'suitable', 'createdAt', 'updatedAt'], axis=1)
    y = df['suitable']
    return X, y

def train_model():
    """
    Train and save the machine learning model and scaler.
    """
    X, y = load_data()
    
    # Define preprocessing for numerical and categorical data
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Define the model pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Define parameter grid for GridSearchCV
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
    
    # Train the model
    print("Starting model training...")
    grid_search.fit(X, y)
    
    # Evaluate the model
    y_pred = grid_search.predict(X)
    print("Classification Report:")
    print(classification_report(y, y_pred))
    
    # Save the best model and scaler
    joblib.dump(grid_search.best_estimator_, 'models/suitability_model.pkl')
    print("Model saved to models/suitability_model.pkl")
    
    # Save the preprocessor (scaler and encoder)
    joblib.dump(grid_search.best_estimator_.named_steps['preprocessor'], 'models/scaler.pkl')
    print("Scaler saved to models/scaler.pkl")

if __name__ == "__main__":
    train_model()