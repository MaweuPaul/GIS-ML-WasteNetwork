import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import traceback
import eventlet

def emit_progress(session_id, message, socketio):
    try:
        if socketio:
            socketio.emit('progress_update', {'session_id': session_id, 'message': message}, room=session_id)
            print(f"Emitted progress: {message}")
            eventlet.sleep(0)
        else:
            print("SocketIO instance not found. Progress message:", message)
    except Exception as e:
        print(f"Failed to emit progress message: {e}")

def emit_error(session_id, message, socketio):
    try:
        if socketio:
            socketio.emit('task_error', {'session_id': session_id, 'message': message}, room=session_id)
            print(f"Emitted error: {message}")
            eventlet.sleep(0)
        else:
            print("SocketIO instance not found. Error message:", message)
    except Exception as e:
        print(f"Failed to emit error message: {e}")
def create_visualizations(X, y, y_pred, feature_importance, model, X_test, y_test, session_id, output_dir):
    # Change from seaborn style to a built-in matplotlib style
    plt.style.use('default')  # or use 'classic', 'bmh', 'ggplot', etc.
    viz_paths = {}
    
    try:
        # 1. Feature Importance with Enhanced Styling
        plt.figure(figsize=(12, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance, 
                    palette='viridis')
        plt.title('Criteria Importance in Suitability Prediction', fontsize=14, pad=20)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Criteria', fontsize=12)
        plt.tight_layout()
        feature_importance_path = os.path.join(output_dir, f'feature_importance_{session_id}.png')
        plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
        plt.close()
        viz_paths['feature_importance'] = feature_importance_path

        # 2. Correlation Heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = X.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    fmt='.2f', square=True)
        plt.title('Correlation Between Suitability Criteria', fontsize=14, pad=20)
        plt.tight_layout()
        correlation_path = os.path.join(output_dir, f'correlation_matrix_{session_id}.png')
        plt.savefig(correlation_path, dpi=300, bbox_inches='tight')
        plt.close()
        viz_paths['correlation_matrix'] = correlation_path

        # 3. Prediction Error Distribution
        plt.figure(figsize=(10, 6))
        errors = y_test - y_pred
        sns.histplot(errors, kde=True)
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        plt.title('Distribution of Prediction Errors', fontsize=14, pad=20)
        plt.xlabel('Prediction Error', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        error_dist_path = os.path.join(output_dir, f'error_distribution_{session_id}.png')
        plt.savefig(error_dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        viz_paths['error_distribution'] = error_dist_path

        # 4. Enhanced Actual vs Predicted Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
        
        # Add trend line
        z = np.polyfit(y_test, y_pred, 1)
        p = np.poly1d(z)
        plt.plot(y_test, p(y_test), "b-", alpha=0.5, label='Trend Line')
        
        plt.title('Actual vs Predicted Suitability Scores', fontsize=14, pad=20)
        plt.xlabel('Actual Suitability', fontsize=12)
        plt.ylabel('Predicted Suitability', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        actual_vs_pred_path = os.path.join(output_dir, f'actual_vs_predicted_{session_id}.png')
        plt.savefig(actual_vs_pred_path, dpi=300, bbox_inches='tight')
        plt.close()
        viz_paths['actual_vs_predicted'] = actual_vs_pred_path

        # 5. Suitability Distribution
        plt.figure(figsize=(12, 6))
        plt.hist(y_test, bins=30, alpha=0.5, label='Actual', density=True)
        plt.hist(y_pred, bins=30, alpha=0.5, label='Predicted', density=True)
        plt.title('Distribution of Actual vs Predicted Suitability Scores', 
                fontsize=14, pad=20)
        plt.xlabel('Suitability Score', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend()
        dist_path = os.path.join(output_dir, f'suitability_distribution_{session_id}.png')
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        viz_paths['suitability_distribution'] = dist_path

        return viz_paths

    except Exception as e:
        print(f"Error in create_visualizations: {str(e)}")
        traceback.print_exc()  # Print the full traceback for debugging
        return viz_paths
def train_model(csv_path, session_id=None, socketio=None):
    try:
        emit_progress(session_id, "🚀 Starting model training process...", socketio)
        
        # Load the data
        emit_progress(session_id, "📊 Loading and preparing data...", socketio)
        df = pd.read_csv(csv_path)
        
        # Define features and target
        feature_columns = ['River_b', 'Road_b', 'Settlem', 'Soil', 'Protect', 'Land_U', 'Slope']
        target_column = 'Total_Suit'
        
        # Prepare X (features) and y (target)
        X = df[feature_columns]
        y = df[target_column]
        
        # Split the data
        emit_progress(session_id, "🔄 Splitting data into training and testing sets...", socketio)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train the model
        emit_progress(session_id, "🤖 Training Random Forest model...", socketio)
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        emit_progress(session_id, "📈 Making predictions and evaluating model...", socketio)
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Create output directory
        output_dir = os.path.join(os.getcwd(), 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model and scaler
        model_path = os.path.join(output_dir, f'model_{session_id}.joblib')
        scaler_path = os.path.join(output_dir, f'scaler_{session_id}.joblib')
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        # Create visualizations
        emit_progress(session_id, "🎨 Creating detailed visualizations...", socketio)
        viz_paths = create_visualizations(
            X=X, 
            y=y, 
            y_pred=y_pred_test,
            feature_importance=feature_importance,
            model=model,
            X_test=X_test,
            y_test=y_test,
            session_id=session_id,
            output_dir=output_dir
        )

        # Prepare results
        results = {
            'model_path': model_path,
            'scaler_path': scaler_path,
            'visualizations': viz_paths,
            'metrics': {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2
            },
            'feature_importance': feature_importance.to_dict('records'),
            'summary_stats': {
                'mean_actual': y_test.mean(),
                'mean_predicted': y_pred_test.mean(),
                'std_actual': y_test.std(),
                'std_predicted': y_pred_test.std(),
                'min_actual': y_test.min(),
                'max_actual': y_test.max()
            }
        }

        return results

    except Exception as e:
        emit_error(session_id, f"❌ Error in train_model: {str(e)}", socketio)
        emit_error(session_id, f"Traceback: {traceback.format_exc()}", socketio)
        return None

