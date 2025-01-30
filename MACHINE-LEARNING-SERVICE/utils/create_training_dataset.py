import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import rasterio
import os
import traceback
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patheffects as pe
import datetime
import eventlet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

# Import the suitability mapping functions from spatial_operations.py
from utils.suitability_mapping import (
    river_suitability_mapping,
    road_suitability_mapping,
    settlement_suitability_mapping,
    protectedarea_suitability_mapping
)
from models.train_model import train_model

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

def calculate_total_suitability(row, weights=None):
    """Calculate weighted suitability score for a row"""
    if weights is None:        # Adjusted weights to balance criteria
        weights = {
            'River_b': 25.0,  # Reduced weight
            'Road_b': 25.0,   # Increased weight
            'Settlem': 20.0,  # Increased weight
            'Soil': 10.0,
            'Protect': 10.0,
            'Land_U': 5.0,
            'Slope': 5.0
        }
    
    # Convert weights to decimals (divide by 100)
    weights = {k: v/100 for k, v in weights.items()}
    
    # Get values for each criterion
    values = {
        'River_b': row['River_buffer'] if 'River_buffer' in row else row['River_b'],
        'Road_b': row['Road_buffer'] if 'Road_buffer' in row else row['Road_b'],
        'Protect': row['Protected_Areas_buffer'] if 'Protected_Areas_buffer' in row else row['Protect'],
        'Settlem': row['Settlement_buffer'] if 'Settlement_buffer' in row else row['Settlem'],
        'Slope': row['Slope'],
        'Land_U': row['Land_U'],
        'Soil': row['Soil']
    }
    
    # Calculate weighted sum
    total_score = 0
    valid_weights_sum = 0
    
    for criterion, weight in weights.items():
        value = values.get(criterion)
        if pd.notna(value):  # Only include non-null values
            total_score += value * weight
            valid_weights_sum += weight
    
    # Normalize by sum of valid weights
    if valid_weights_sum > 0:
        total_score = total_score / valid_weights_sum * 5  # Scale to 1-5 range
    
    return round(total_score, 2)

def clean_and_process_training_data(training_gdf, session_id, socketio):
    try:
        emit_progress(session_id, "Cleaning and processing training data...", socketio)
        
        # Create a copy to avoid modifying the original
        cleaned_gdf = training_gdf.copy()
        
        # Rename columns if needed
        column_mapping = {
            'River_buffer': 'River_b',
            'Road_buffer': 'Road_b',
            'Protected_Areas_buffer': 'Protect',
            'Settlement_buffer': 'Settlem'
        }
        cleaned_gdf = cleaned_gdf.rename(columns=column_mapping)
        
        # Fill NaN values with appropriate defaults
        default_values = {
            'River_b': 5,    # Highly suitable for areas far from rivers
            'Road_b': 1,     # Not suitable for areas far from roads
            'Settlem': 4,    # Suitable for areas far from settlements
            'Soil': 3,       # Moderate for unknown soil type
            'Protect': 5,    # Highly suitable for areas far from protected areas
            'Land_U': 3,     # Moderate for unknown land use
            'Slope': 3       # Moderate for unknown slope
        }
        
        cleaned_gdf = cleaned_gdf.fillna(default_values)
        
        # Calculate total suitability score with specific weights
        weights = {
            'River_b': 25.0,
            'Road_b': 25.0,
            'Settlem': 20.0,
            'Soil': 10.0,
            'Protect': 10.0,
            'Land_U': 5.0,
            'Slope': 5.0
        }
        cleaned_gdf['Total_Suit'] = cleaned_gdf.apply(lambda row: calculate_total_suitability(row, weights), axis=1)
        
        # Add suitability classification
        def classify_suitability(score):
            if score <= 1:
                return 'Not Suitable'
            elif score <= 2:
                return 'Less Suitable'
            elif score <= 3:
                return 'Moderately Suitable'
            elif score <= 4:
                return 'Suitable'
            else:
                return 'Highly Suitable'
        
        cleaned_gdf['Suitability_Class'] = cleaned_gdf['Total_Suit'].apply(classify_suitability)
        
        # Round numeric columns to 2 decimal places
        numeric_columns = cleaned_gdf.select_dtypes(include=[np.number]).columns
        cleaned_gdf[numeric_columns] = cleaned_gdf[numeric_columns].round(2)
        
        # Keep only necessary columns and reorder them
        columns_to_keep = [
            'geometry', 'x', 'y', 
            'River_b', 'Road_b', 'Settlem', 'Soil', 
            'Protect', 'Land_U', 'Slope',
            'Total_Suit', 'Suitability_Class'
        ]
        cleaned_gdf = cleaned_gdf[columns_to_keep]
        
        # Add weight information to the output
        emit_progress(session_id, "\nWeights used in analysis:", socketio)
        for criterion, weight in weights.items():
            emit_progress(session_id, f"{criterion}: {weight}%", socketio)
        
        emit_progress(session_id, "Data cleaning and processing completed.", socketio)
        return cleaned_gdf
        
    except Exception as e:
        emit_error(session_id, f"Error in cleaning training data: {str(e)}", socketio)
        return None

# Define buffer distances for each feature type
buffer_distances = {
    'River': [300, 1000, 1500, 2000],
    'Road': [400, 800, 1000, 1200],
    'Settlement': [400, 900, 1500, 2100],
    'Protected_Areas': [300, 1000, 1500, 2000, 2500]
}

def calculate_distance_to_feature(point, feature_geom):
    """Calculate the minimum distance from a point to a feature"""
    return point.distance(feature_geom)

def refine_model(model, X_train, X_test, y_train, y_test, session_id, socketio):
    """
    Refine the Random Forest model through hyperparameter tuning and cross-validation
    """
   
    
    emit_progress(session_id, "\n🔄 Starting model refinement process...", socketio)
    
    try:
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            scoring='r2',
            verbose=1
        )
        
        emit_progress(session_id, "Performing grid search for optimal parameters...", socketio)
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Evaluate refined model
        train_score = r2_score(y_train, best_model.predict(X_train))
        test_score = r2_score(y_test, best_model.predict(X_test))
        
        emit_progress(session_id, "\n📊 Model Refinement Results:", socketio)
        emit_progress(session_id, f"Best Parameters: {grid_search.best_params_}", socketio)
        emit_progress(session_id, f"Training R² Score: {train_score:.4f}", socketio)
        emit_progress(session_id, f"Testing R² Score: {test_score:.4f}", socketio)
        
        # Check for overfitting
        if train_score - test_score > 0.1:
            emit_progress(session_id, "\n⚠️ Warning: Model might be overfitting", socketio)
            
            # Additional refinement for overfitting
            best_model = handle_overfitting(best_model, X_train, y_train, X_test, y_test)
        
        return best_model, {
            'best_params': grid_search.best_params_,
            'train_score': train_score,
            'test_score': test_score
        }
        
    except Exception as e:
        emit_error(session_id, f"Error in model refinement: {str(e)}", socketio)
        return None, None

def handle_overfitting(model, X_train, y_train, X_test, y_test):
    """
    Handle overfitting by adjusting model parameters
    """
    # Increase regularization
    model.set_params(
        max_depth=min(model.max_depth - 5 if model.max_depth else 15, 10),
        min_samples_leaf=max(model.min_samples_leaf, 2),
        min_samples_split=max(model.min_samples_split, 5)
    )
    
    # Retrain with adjusted parameters
    model.fit(X_train, y_train)
    
    return model
def create_training_dataset(nyeri_gdf, buffer_sets, raster_criteria, n_points=10000, session_id=None, socketio=None):
    try:
        emit_progress(session_id, "🚀 Initializing training dataset creation process...", socketio)
        
        # Verify input types
        if not isinstance(nyeri_gdf, gpd.GeoDataFrame):
            raise ValueError(f"nyeri_gdf must be a GeoDataFrame, got {type(nyeri_gdf)}")
        
        emit_progress(session_id, "\n🗺️ Getting study area boundaries...", socketio)
        bounds = nyeri_gdf.total_bounds
        study_area_polygon = nyeri_gdf.unary_union
        
        # Generate points in batches
        batch_size = min(1000, n_points)
        points = []
        point_data = []
        
        emit_progress(session_id, "\n📍 Starting point generation process...", socketio)
        emit_progress(session_id, f"Target: {n_points} points", socketio)
        
        progress_interval = max(1, n_points // 20)  # Update every 5%
        
        while len(points) < n_points:
            # Generate random points within the bounds
            xs = np.random.uniform(bounds[0], bounds[2], batch_size)
            ys = np.random.uniform(bounds[1], bounds[3], batch_size)
            batch_points = [Point(x, y) for x, y in zip(xs, ys)]
            
            # Filter points within study area
            valid_points = [p for p in batch_points if study_area_polygon.contains(p)]
            
            for point in valid_points:
                if len(points) >= n_points:
                    break
                    
                points.append(point)
                point_info = {
                    'geometry': point,
                    'x': point.x,
                    'y': point.y
                }
                
                # Get buffer zone values
                for feature_name, buffer_list in buffer_sets.items():
                    if not isinstance(buffer_list, list):
                        continue
                    
                    # Check each buffer zone
                    for i, buffer_gdf in enumerate(buffer_list):
                        if any(buffer_gdf.geometry.contains(point)):
                            point_info[f'{feature_name}_buffer'] = i + 1
                            break
                    else:
                        point_info[f'{feature_name}_buffer'] = len(buffer_list) + 1
                
                point_data.append(point_info)
                
                if len(points) % progress_interval == 0:
                    progress_percent = (len(points) / n_points) * 100
                    emit_progress(session_id, f"⏳ Generated {len(points)}/{n_points} points ({progress_percent:.1f}%)", socketio)
        
        emit_progress(session_id, "\n📊 Creating spatial database with generated points...", socketio)
        training_gdf = gpd.GeoDataFrame(point_data, geometry='geometry', crs=nyeri_gdf.crs)
        
        emit_progress(session_id, "\n🔍 Sampling raster criteria values...", socketio)
        for raster_name, raster_path in raster_criteria.items():
            try:
                emit_progress(session_id, f"  • Processing {raster_name}...", socketio)
                with rasterio.open(raster_path) as src:
                    coords = [(p.x, p.y) for p in points]
                    values = [val[0] for val in src.sample(coords)]
                    training_gdf[raster_name] = values
            except Exception as e:
                emit_error(session_id, f"⚠️ Error processing {raster_name}: {str(e)}", socketio)
                training_gdf[raster_name] = 3  # Default moderate suitability
        
        emit_progress(session_id, "\n🧹 Cleaning and standardizing data...", socketio)
        # Clean and standardize column names
        column_mapping = {
            'River_buffer': 'River_b',
            'Road_buffer': 'Road_b',
            'Protected_Areas_buffer': 'Protect',
            'Settlement_buffer': 'Settlem',
            'landuse': 'Land_U',
            'land_use': 'Land_U',
            'Land_Use': 'Land_U',
            'LandUse': 'Land_U',
            'LANDUSE': 'Land_U',
            'slope': 'Slope',
            'SLOPE': 'Slope'
        }
        
        training_gdf = training_gdf.rename(columns=column_mapping)
        
        # Define value ranges for each criterion
        value_ranges = {
            'River_b': (1, 5),
            'Road_b': (1, 5),
            'Settlem': (1, 5),
            'Soil': (1, 5),
            'Protect': (1, 5),
            'Land_U': (1, 5),
            'Slope': (1, 5)
        }
        
        # Define weights (sum to 100%)
        weights = {
            'River_b': 25/100,
            'Road_b': 25/100,
            'Settlem': 20/100,
            'Soil': 10/100,
            'Protect': 10/100,
            'Land_U': 5/100,
            'Slope': 5/100
        }
        
        # Fill NaN values with appropriate defaults
        default_values = {
            'River_b': 3,
            'Road_b': 3,
            'Settlem': 3,
            'Soil': 3,
            'Protect': 3,
            'Land_U': 3,
            'Slope': 3
        }
        
        # Check for missing columns and add them with default values
        criteria_columns = ['River_b', 'Road_b', 'Settlem', 'Soil', 'Protect', 'Land_U', 'Slope']
        for col, default_val in default_values.items():
            if col not in training_gdf.columns:
                emit_progress(session_id, f"⚠️ Adding missing column {col} with default value {default_val}", socketio)
                training_gdf[col] = default_val
        
        training_gdf = training_gdf.fillna(default_values)
        
        # Calculate weighted suitability with proper normalization
        total_suit = 0
        for col in criteria_columns:
            if col in training_gdf.columns:
                # Clip values to their proper ranges
                min_val, max_val = value_ranges[col]
                training_gdf[col] = training_gdf[col].clip(min_val, max_val)
                
                # Normalize to 0-1 range based on actual range
                normalized_values = (training_gdf[col] - min_val) / (max_val - min_val)
                
                # Apply weight and add to total
                total_suit += normalized_values * weights[col]
        
        # Scale back to 1-5 range
        training_gdf['Total_Suit'] = 1 + (total_suit * 4)
        training_gdf['Total_Suit'] = training_gdf['Total_Suit'].round(2)
        
        # Classify suitability
        def classify_suitability(score):
            if score <= 2:
                return 'Not Suitable'
            elif score <= 2.75:
                return 'Less Suitable'
            elif score <= 3.5:
                return 'Moderately Suitable'
            elif score <= 4.25:
                return 'Suitable'
            else:
                return 'Highly Suitable'
        
        training_gdf['Suitability_Class'] = training_gdf['Total_Suit'].apply(classify_suitability)
        
        # Round numeric columns
        numeric_columns = training_gdf.select_dtypes(include=[np.number]).columns
        training_gdf[numeric_columns] = training_gdf[numeric_columns].round(2)
        
        # Save outputs
        emit_progress(session_id, "\n💾 Saving outputs...", socketio)
        output_dir = os.path.join(os.getcwd(), 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"training_points_{session_id}_{timestamp}"
        
        plot_path = os.path.join(output_dir, f"{base_filename}.png")
        csv_path = os.path.join(output_dir, f"{base_filename}.csv")
        
        training_gdf.to_csv(csv_path, index=False)
        
        # Train the model with the created CSV
        emit_progress(session_id, "\n🤖 Starting model training...", socketio)
        model_results = train_model(csv_path, session_id, socketio)
        

        if model_results and 'model' in model_results:
            # Refine the model
            emit_progress(session_id, "\n🔄 Starting model refinement...", socketio)
            refined_model, refinement_results = refine_model(
                model_results['model'],
                model_results['X_train'],
                model_results['X_test'],
                model_results['y_train'],
                model_results['y_test'],
                session_id,
                socketio
            )
            
            if refined_model is not None:
                model_results['model'] = refined_model
                model_results['refinement_results'] = refinement_results
                emit_progress(session_id, "\n✅ Model refinement completed successfully!", socketio)
            else:
                emit_error(session_id, "❌ Model refinement failed", socketio)
        
        return training_gdf, plot_path, csv_path, model_results
    
    except Exception as e:
        emit_error(session_id, f"❌ Error in create_training_dataset: {str(e)}", socketio)
        emit_error(session_id, f"Traceback: {traceback.format_exc()}", socketio)
        return None, None, None, None