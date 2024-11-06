import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Tuple, Dict, Any, List
import joblib
import os
from datetime import datetime
import json

class LandfillSuitabilityML:
    def __init__(self, session_id, socketio):
        """
        Initialize the ML analysis class
        
        Args:
            session_id: Unique session identifier
            socketio: SocketIO instance for real-time updates
        """
        self.session_id = session_id
        self.socketio = socketio
        self.logger = logging.getLogger(__name__)
        # Change to RandomForestRegressor
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = []
        self.output_dir = os.path.join('output', f'session_{session_id}')
        os.makedirs(self.output_dir, exist_ok=True)

    def prepare_training_data(self, features_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from feature layers
        """
        try:
            self.emit_progress("Preparing training data...")
            
            feature_arrays = []
            self.feature_names = []
            
            for name, feature_data in features_dict.items():
                if feature_data is not None:
                    flat_data = feature_data.ravel()
                    valid_mask = ~np.isnan(flat_data)
                    feature_arrays.append(flat_data[valid_mask].reshape(-1, 1))
                    self.feature_names.append(name)
            
            X = np.hstack(feature_arrays)
            X_scaled = self.scaler.fit_transform(X)
            
            # Create target variable using weighted criteria
            y = self.create_target_variable(X)
            
            self.emit_progress(f"Prepared training data with shape: {X.shape}")
            self.save_feature_statistics(X, y)
            
            return X_scaled, y
            
        except Exception as e:
            self.emit_error(f"Error preparing training data: {str(e)}")
            raise

    def create_target_variable(self, X: np.ndarray) -> np.ndarray:
        """
        Create target variable using weighted criteria
        """
        try:
            # Define weights
            weights = {
                'River': 0.3654,
                'Road': 0.2586,
                'Settlement': 0.1797,
                'Soil': 0.0924,
                'Protected_Areas': 0.0475,
                'Land_Use': 0.0330,
                'Slope': 0.0234
            }

            # Calculate weighted suitability scores
            weighted_scores = np.zeros(X.shape[0])

            for feature_name, weight in weights.items():
                if feature_name in self.feature_names:
                    feature_idx = self.feature_names.index(feature_name)
                    feature_scores = X[:, feature_idx]
                    
                    # Apply feature-specific criteria and weights
                    if feature_name == 'River':
                        weighted_scores += (feature_scores >= 3) * weight * feature_scores
                    elif feature_name == 'Road':
                        weighted_scores += (feature_scores >= 4) * weight * feature_scores
                    elif feature_name == 'Settlement':
                        weighted_scores += (feature_scores >= 3) * weight * feature_scores
                    elif feature_name == 'Soil':
                        weighted_scores += (feature_scores >= 3) * weight * feature_scores
                    elif feature_name == 'Protected_Areas':
                        weighted_scores += (feature_scores >= 4) * weight * feature_scores
                    elif feature_name == 'Land_Use':
                        weighted_scores += (feature_scores >= 4) * weight * feature_scores
                    elif feature_name == 'Slope':
                        weighted_scores += (feature_scores >= 4) * weight * feature_scores

            # Create mandatory criteria mask
            mandatory_features = ['River', 'Settlement', 'Soil', 'Protected_Areas']
            mandatory_criteria = np.ones(X.shape[0], dtype=bool)
            
            for feature in mandatory_features:
                if feature in self.feature_names:
                    feature_idx = self.feature_names.index(feature)
                    if feature in ['River', 'Settlement', 'Soil']:
                        mandatory_criteria &= (X[:, feature_idx] >= 3)
                    elif feature == 'Protected_Areas':
                        mandatory_criteria &= (X[:, feature_idx] >= 4)

            # Classify based on weighted scores and mandatory criteria
            classification = np.zeros(X.shape[0], dtype=int)
            
            # Calculate thresholds
            total_possible_score = sum(weights.values()) * 5
            high_threshold = 0.8 * total_possible_score
            medium_threshold = 0.6 * total_possible_score

            # Classify valid locations
            classification[mandatory_criteria & (weighted_scores >= high_threshold)] = 2
            classification[mandatory_criteria & 
                         (weighted_scores >= medium_threshold) & 
                         (weighted_scores < high_threshold)] = 1

            self.save_classification_stats(classification, weighted_scores)
            return classification

        except Exception as e:
            self.emit_error(f"Error creating target variable: {str(e)}")
            raise

    def train_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        try:
            self.emit_progress("Starting model training...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
    
            # Perform grid search for hyperparameter tuning
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }
            
            grid_search = GridSearchCV(
                self.model, param_grid, cv=5, scoring='r2', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            # Use best model
            self.model = grid_search.best_estimator_
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate regression metrics
            metrics = {
                'r2_score': r2_score(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'feature_importance': self.get_feature_importance()
            }
            
            # Create and save plots
            self.create_evaluation_plots(X_test, y_test, y_pred)
            
            # Save model and metrics
            self.save_results(metrics)
            
            return metrics
            
        except Exception as e:
            self.emit_error(f"Error in model training: {str(e)}")
            raise

    def predict_suitability(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict suitability for new data
        """
        try:
            # Prepare prediction data
            X_pred = []
            for feature_name in self.feature_names:
                if feature_name in features_dict:
                    X_pred.append(features_dict[feature_name].ravel())
            
            X_pred = np.column_stack(X_pred)
            X_pred_scaled = self.scaler.transform(X_pred)
            
            # Make predictions
            predictions = self.model.predict(X_pred_scaled)
            
            # Reshape predictions to match original raster shape
            prediction_raster = predictions.reshape(
                features_dict[self.feature_names[0]].shape
            )
            
            return prediction_raster
            
        except Exception as e:
            self.emit_error(f"Error in prediction: {str(e)}")
            raise

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance.to_dict('records')

    def create_evaluation_plots(self, X_test: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray):
        try:
            # Predicted vs Actual
            plt.figure(figsize=(10, 8))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Predicted vs Actual Values')
            plt.savefig(os.path.join(self.output_dir, 'prediction_scatter.png'))
            plt.close()
    
            # Feature Importance
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(12, 6))
            sns.barplot(x='importance', y='feature', data=importance)
            plt.title('Feature Importance')
            plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'))
            plt.close()
    
            # Residuals Plot
            residuals = y_test - y_pred
            plt.figure(figsize=(10, 8))
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residuals Plot')
            plt.axhline(y=0, color='r', linestyle='--')
            plt.savefig(os.path.join(self.output_dir, 'residuals.png'))
            plt.close()
    
        except Exception as e:
            self.emit_error(f"Error creating evaluation plots: {str(e)}")

    def save_results(self, metrics: Dict[str, Any]):
        """Save model and results"""
        try:
            # Save model
            model_path = os.path.join(self.output_dir, 'model.joblib')
            joblib.dump(self.model, model_path)
            
            # Save scaler
            scaler_path = os.path.join(self.output_dir, 'scaler.joblib')
            joblib.dump(self.scaler, scaler_path)
            
            # Save metrics
            metrics_path = os.path.join(self.output_dir, 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            self.emit_progress("Model and results saved successfully")
            
        except Exception as e:
            self.emit_error(f"Error saving results: {str(e)}")

    def save_feature_statistics(self, X: np.ndarray, y: np.ndarray):
        """Save feature statistics"""
        try:
            stats = {
                'feature_stats': {},
                'target_distribution': {}
            }
            
            for i, feature in enumerate(self.feature_names):
                stats['feature_stats'][feature] = {
                    'mean': float(np.mean(X[:, i])),
                    'std': float(np.std(X[:, i])),
                    'min': float(np.min(X[:, i])),
                    'max': float(np.max(X[:, i]))
                }
            
            unique, counts = np.unique(y, return_counts=True)
            stats['target_distribution'] = dict(zip(unique.tolist(), counts.tolist()))
            
            stats_path = os.path.join(self.output_dir, 'feature_stats.json')
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=4)
                
        except Exception as e:
            self.emit_error(f"Error saving feature statistics: {str(e)}")

    def save_classification_stats(self, classification: np.ndarray, weighted_scores: np.ndarray):
        """Save classification statistics"""
        try:
            stats = {
                'class_distribution': dict(zip(
                    *np.unique(classification, return_counts=True)
                )),
                'weighted_scores_stats': {
                    'mean': float(np.mean(weighted_scores)),
                    'std': float(np.std(weighted_scores)),
                    'min': float(np.min(weighted_scores)),
                    'max': float(np.max(weighted_scores))
                }
            }
            
            stats_path = os.path.join(self.output_dir, 'classification_stats.json')
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=4)
                
        except Exception as e:
            self.emit_error(f"Error saving classification statistics: {str(e)}")

    def emit_progress(self, message: str):
        """Emit progress message"""
        if self.socketio:
            self.socketio.emit('progress_update', {
                'session_id': self.session_id,
                'message': message
            })
        self.logger.info(message)

    def emit_error(self, message: str):
        """Emit error message"""
        if self.socketio:
            self.socketio.emit('error_update', {
                'session_id': self.session_id,
                'message': message
            })
        self.logger.error(message)