import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box, Polygon, LineString, MultiLineString
import rasterio
from rasterio.windows import from_bounds
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import dask.dataframe as dd
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask
from flask_socketio import SocketIO
import eventlet
import traceback
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

def emit_progress(session_id, message, socketio):
    try:
        if socketio:
            socketio.emit('progress_update', {'session_id': session_id, 'message': message}, room=session_id)
            logging.info(f"Emitted progress: {message}")
            eventlet.sleep(0)
        else:
            logging.warning("SocketIO instance not found. Progress message: " + message)
    except Exception as e:
        logging.error(f"Failed to emit progress message: {e}")

def emit_error(session_id, message, socketio):
    try:
        if socketio:
            socketio.emit('task_error', {'session_id': session_id, 'message': message}, room=session_id)
            logging.error(f"Emitted error: {message}")
            eventlet.sleep(0)
        else:
            logging.warning("SocketIO instance not found. Error message: " + message)
    except Exception as e:
        logging.error(f"Failed to emit error message: {e}")

def emit_sample_data(session_id, data_type, data, socketio):
    try:
        if socketio:
            socketio.emit('sample_data', {'session_id': session_id, 'type': data_type, 'data': data}, room=session_id)
            logging.info(f"Emitted sample data: {data_type}")
            eventlet.sleep(0)
        else:
            logging.warning(f"SocketIO instance not found. Sample data: {data_type}")
    except Exception as e:
        logging.error(f"Failed to emit sample data: {e}")

def clean_geometries(gdf, session_id, socketio):
    emit_progress(session_id, "Cleaning geometries: Removing null geometries", socketio)
    gdf = gdf[gdf.geometry.notna()]
    emit_progress(session_id, f"Geometries after removing nulls: {len(gdf)}", socketio)
    
    emit_progress(session_id, "Cleaning geometries: Buffering geometries by 0 to fix invalid geometries", socketio)
    gdf['geometry'] = gdf.geometry.buffer(0)
    
    emit_progress(session_id, "Cleaning geometries: Removing invalid geometries", socketio)
    gdf = gdf[gdf.geometry.is_valid]
    emit_progress(session_id, f"Geometries after validation: {len(gdf)}", socketio)
    
    emit_progress(session_id, "Cleaning geometries: Converting geometries to Polygon objects where necessary", socketio)
    gdf['geometry'] = gdf.geometry.apply(
        lambda geom: Polygon(geom) if isinstance(geom, (LineString, MultiLineString)) else geom
    )
    
    emit_progress(session_id, "Cleaning geometries completed", socketio)
    return gdf

def create_grid(nyeri_gdf, cell_size, session_id, socketio):
    try:
        emit_progress(session_id, "Creating grid: Calculating total bounds", socketio)
        bounds = nyeri_gdf.total_bounds
        x_min, y_min, x_max, y_max = bounds
        emit_progress(session_id, f"Grid bounds: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}", socketio)
        
        emit_progress(session_id, "Creating grid: Generating coordinate ranges", socketio)
        x_coords = np.arange(x_min, x_max + cell_size, cell_size)
        y_coords = np.arange(y_min, y_max + cell_size, cell_size)

        emit_progress(session_id, "Creating grid: Generating grid cells", socketio)
        grid_cells = []
        for x in x_coords[:-1]:
            for y in y_coords[:-1]:
                cell = box(x, y, x + cell_size, y + cell_size)
                if nyeri_gdf.intersects(cell).any() and cell.is_valid:
                    grid_cells.append(cell)
        
        if not grid_cells:
            raise ValueError("No valid grid cells created")
        
        emit_progress(session_id, f"Number of grid cells created: {len(grid_cells)}", socketio)
        grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs=nyeri_gdf.crs)
        grid_gdf['cell_id'] = range(len(grid_gdf))
        
        emit_progress(session_id, "Grid creation completed successfully", socketio)
        return grid_gdf

    except Exception as e:
        error_message = f"Error in create_grid: {str(e)}"
        emit_error(session_id, error_message, socketio)
        logging.error(error_message)
        logging.error(traceback.format_exc())
        return gpd.GeoDataFrame(columns=['geometry', 'cell_id'], crs=nyeri_gdf.crs)

def extract_layer_data(grid_gdf, layers, session_id, socketio):
    def process_layer(layer_name, layer_info):
        if layer_name.lower() == 'total_suitability':
            logging.info(f"Skipping layer '{layer_name}' as it is a composite layer.")
            return [np.nan] * len(grid_gdf)

        try:
            with rasterio.open(layer_info['path']) as src:
                values = []
                for idx, cell in enumerate(grid_gdf.geometry):
                    try:
                        cell_bounds = cell.bounds
                        window = from_bounds(*cell_bounds, src.transform)
                        cell_data = src.read(1, window=window)
                        if cell_data.size > 0:
                            mean_val = np.nanmean(cell_data)
                            values.append(mean_val)
                        else:
                            values.append(np.nan)
                    except Exception as cell_e:
                        logging.error(f"Error processing cell {idx} for layer '{layer_name}': {cell_e}")
                        values.append(np.nan)
                return values
        except Exception as e:
            emit_error(session_id, f"Error processing layer '{layer_name}': {str(e)}", socketio)
            logging.error(f"Error processing layer '{layer_name}': {str(e)}")
            logging.error(traceback.format_exc())
            return [np.nan] * len(grid_gdf)

    meta = pd.DataFrame({layer_name: pd.Series(dtype='float') for layer_name in layers.keys() if layer_name.lower() != 'total_suitability'})
    
    dask_grid = dd.from_pandas(grid_gdf.reset_index(drop=True), npartitions=4)
    
    emit_progress(session_id, "Starting data extraction for layers", socketio)
    results = dask_grid.map_partitions(
        lambda df: pd.DataFrame({
            layer_name: process_layer(layer_name, layer_info)
            for layer_name, layer_info in layers.items()
            if layer_name.lower() != 'total_suitability'
        }),
        meta=meta
    ).compute()
    
    results = results.reset_index(drop=True)
    updated_grid_gdf = pd.concat([grid_gdf.reset_index(drop=True), results], axis=1)
    
    emit_sample_data(session_id, "extracted_layer_data", updated_grid_gdf.head().to_dict('records'), socketio)
    
    logging.info("Data extraction for layers completed successfully.")
    emit_progress(session_id, "Data extraction for layers completed successfully", socketio)
    
    return gpd.GeoDataFrame(updated_grid_gdf, geometry='geometry', crs=grid_gdf.crs)

def save_extracted_data(grid_gdf, session_id):
    if grid_gdf is None:
        raise ValueError("grid_gdf is None in save_extracted_data")
    os.makedirs('output', exist_ok=True)
    excel_path = os.path.join('output', f'extracted_data_session_{session_id}.xlsx')
    
    try:
        with pd.ExcelWriter(excel_path) as writer:
            grid_data = grid_gdf.copy()
            grid_data['geometry'] = grid_data['geometry'].apply(lambda geom: geom.wkt if geom is not None else 'INVALID')
            grid_data.to_excel(writer, sheet_name='Extracted Data', index=False)
            
            nan_report = grid_gdf.isna().sum().reset_index()
            nan_report.columns = ['Column', 'NaN Count']
            nan_report['NaN Percentage'] = nan_report['NaN Count'] / len(grid_gdf) * 100
            nan_report.to_excel(writer, sheet_name='NaN Report', index=False)
        
        logging.info(f"Extracted data saved to: {excel_path}")
        emit_progress(session_id, f"Extracted data saved to: {excel_path}", socketio)
        return excel_path
    except Exception as e:
        emit_error(session_id, f"Error saving extracted data: {str(e)}", socketio)
        logging.error(f"Error saving extracted data: {str(e)}")
        logging.error(traceback.format_exc())
        return "Failed to save extracted data"

def preprocess_data(grid_gdf, weights, session_id, socketio):
    continuous_vars = ['Buffer_River', 'Buffer_Road', 'Buffer_ProtectedArea', 'Buffer_Settlement', 'Slope']
    categorical_vars = ['LandUse', 'Geology', 'Soil']

    emit_progress(session_id, "Starting data preprocessing", socketio)

    emit_progress(session_id, "Preprocessing continuous variables", socketio)
    for var in continuous_vars:
        grid_gdf[var] = pd.to_numeric(grid_gdf[var], errors='coerce')
        grid_gdf[var] = grid_gdf[var].interpolate(method='linear', limit_direction='both')
        grid_gdf[var] = grid_gdf[var].fillna(grid_gdf[var].mean())
        logging.info(f"Preprocessed continuous variable: {var}")
    
    emit_progress(session_id, "Preprocessing categorical variables", socketio)
    for var in categorical_vars:
        grid_gdf[var] = grid_gdf[var].fillna('Unknown').astype(str)
        logging.info(f"Preprocessed categorical variable: {var}")
    
    total_columns = len(continuous_vars + categorical_vars)
    grid_gdf['data_completeness'] = grid_gdf[continuous_vars + categorical_vars].notna().sum(axis=1) / total_columns
    emit_progress(session_id, "Calculated data completeness", socketio)
    
    emit_progress(session_id, "Normalizing continuous variables", socketio)
    for var in continuous_vars:
        min_val = grid_gdf[var].min()
        max_val = grid_gdf[var].max()
        if min_val != max_val:
            grid_gdf[var] = 1 + 4 * (grid_gdf[var] - min_val) / (max_val - min_val)
        else:
            grid_gdf[var] = 3  # Arbitrary mid-point if no variation
    
    emit_progress(session_id, "Calculating suitability scores", socketio)
    grid_gdf['Suitability'] = sum(grid_gdf[var] * weights.get(var, 1) for var in continuous_vars)
    emit_progress(session_id, "Data preprocessing completed successfully", socketio)
    
    return grid_gdf

def prepare_data_for_ml(data, is_prediction=False):
    continuous_vars = ['Buffer_River', 'Buffer_Road', 'Buffer_ProtectedArea', 'Buffer_Settlement', 'Slope']
    categorical_vars = ['LandUse', 'Geology', 'Soil']
    
    X = data[continuous_vars + categorical_vars]
    
    if not is_prediction:
        y = data['Suitability']
        return X, y
    else:
        return X

def train_and_evaluate_model(X, y):
    emit_progress(session_id=None, message="Splitting data into training and testing sets", socketio=socketio)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = ['Buffer_River', 'Buffer_Road', 'Buffer_ProtectedArea', 'Buffer_Settlement', 'Slope']
    categorical_features = ['LandUse', 'Geology', 'Soil']

    emit_progress(session_id=None, message="Setting up preprocessing pipeline", socketio=socketio)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    emit_progress(session_id=None, message="Setting up machine learning pipeline", socketio=socketio)
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    emit_progress(session_id=None, message="Training the model", socketio=socketio)
    model.fit(X_train, y_train)
    emit_progress(session_id=None, message="Model training completed", socketio=socketio)

    emit_progress(session_id=None, message="Making predictions on the test set", socketio=socketio)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

    logging.info(f"Model trained. MSE: {mse:.4f}, R2: {r2:.4f}")
    emit_progress(session_id=None, message=f"Model trained. MSE: {mse:.4f}, R2: {r2:.4f}", socketio=socketio)

    return model, mse, r2, cv_scores

def create_analysis_report(model, X, mse, r2, cv_scores, session_id, socketio):
    report = f"Model: {model}\n"
    report += f"Mean Squared Error: {mse:.4f}\n"
    report += f"R-squared Score: {r2:.4f}\n"
    report += f"Cross-validation Scores: {cv_scores}\n"
    report += f"Mean CV Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})\n"
    
    emit_progress(session_id, "Creating analysis report", socketio)
    logging.info("Analysis report created.")
    emit_sample_data(session_id, "analysis_report", {"report": report}, socketio)
    
    return report

def create_visualizations(sampled_points, model, X, session_id, socketio):
    plot_paths = {}
    
    try:
        emit_progress(session_id, "Creating feature importance plot", socketio)
        if hasattr(model.named_steps['regressor'], 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.named_steps['regressor'].feature_importances_
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=feature_importance)
            plt.title('Feature Importance')
            plt.tight_layout()
            feature_importance_path = os.path.join('output', f'feature_importance_session_{session_id}.png')
            plt.savefig(feature_importance_path)
            plt.close()
            plot_paths['Feature Importance'] = feature_importance_path
            emit_progress(session_id, "Feature importance plot created", socketio)
        else:
            logging.warning("Model does not have feature_importances_ attribute. Skipping feature importance plot.")
            emit_progress(session_id, "Model lacks feature_importances_. Skipped feature importance plot", socketio)

        emit_progress(session_id, "Creating scatter plot for Actual vs Predicted Suitability", socketio)
        if 'Suitability' in sampled_points.columns and 'Predicted_Suitability' in sampled_points.columns:
            scatter_data = sampled_points.dropna(subset=['Suitability', 'Predicted_Suitability'])
            if len(scatter_data['Suitability']) > 0 and len(scatter_data['Predicted_Suitability']) > 0:
                plt.figure(figsize=(10, 6))
                plt.scatter(scatter_data['Suitability'], scatter_data['Predicted_Suitability'])
                plt.xlabel('Actual Suitability')
                plt.ylabel('Predicted Suitability')
                plt.title('Actual vs Predicted Suitability')
                plt.tight_layout()
                scatter_plot_path = os.path.join('output', f'actual_vs_predicted_session_{session_id}.png')
                plt.savefig(scatter_plot_path)
                plt.close()
                plot_paths['Actual vs Predicted'] = scatter_plot_path
                emit_progress(session_id, "Scatter plot for Actual vs Predicted Suitability created", socketio)
            else:
                logging.warning("Insufficient data for scatter plot. Skipping scatter plot.")
                emit_progress(session_id, "Insufficient data for scatter plot. Skipped scatter plot", socketio)
        else:
            logging.warning("'Suitability' or 'Predicted_Suitability' column not found in sampled_points. Skipping scatter plot.")
            emit_progress(session_id, "'Suitability' or 'Predicted_Suitability' column missing. Skipped scatter plot", socketio)

    except Exception as e:
        error_message = f"Error in create_visualizations: {str(e)}"
        emit_error(session_id, error_message, socketio)
        logging.error(error_message)
        logging.error(traceback.format_exc())
        plot_paths['error'] = error_message

    return plot_paths

def save_to_excel(grid_gdf, sampled_points, model, session_id):
    excel_path = os.path.join('output', f'processed_data_session_{session_id}.xlsx')
    
    try:
        with pd.ExcelWriter(excel_path) as writer:
            grid_data = grid_gdf.copy()
            grid_data['geometry'] = grid_data['geometry'].apply(lambda geom: geom.wkt if geom is not None else 'INVALID')
            grid_data.to_excel(writer, sheet_name='Full Grid Data', index=False)
            
            sampled_points_data = sampled_points.copy()
            sampled_points_data['geometry'] = sampled_points_data['geometry'].apply(lambda geom: geom.wkt if geom is not None else 'INVALID')
            sampled_points_data.to_excel(writer, sheet_name='Sampled Points', index=False)
            
            emit_progress(session_id, "Saving feature importance to Excel", socketio)
            if hasattr(model.named_steps['regressor'], 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'Feature': grid_gdf.drop(['geometry', 'cell_id', 'Suitability', 'Predicted_Suitability', 'data_completeness'], axis=1, errors='ignore').columns,
                    'Importance': model.named_steps['regressor'].feature_importances_
                }).sort_values('Importance', ascending=False)
                feature_importance.to_excel(writer, sheet_name='Feature Importance', index=False)
            else:
                logging.warning("Model does not have feature_importances_. Skipping feature importance in Excel.")
            
            emit_progress(session_id, "Saving NaN report to Excel", socketio)
            nan_report = grid_gdf.isna().sum().reset_index()
            nan_report.columns = ['Column', 'NaN Count']
            nan_report['NaN Percentage'] = nan_report['NaN Count'] / len(grid_gdf) * 100
            nan_report.to_excel(writer, sheet_name='NaN Report', index=False)
        
        logging.info(f"Processed data saved to Excel: {excel_path}")
        emit_progress(session_id, f"Processed data saved to Excel: {excel_path}", socketio)
        return excel_path
    except Exception as e:
        emit_error(session_id, f"Error saving to Excel: {str(e)}", socketio)
        logging.error(f"Error saving to Excel: {str(e)}")
        logging.error(traceback.format_exc())
        return "Failed to save to Excel"

def create_prediction_map(grid_gdf, nyeri_gdf, session_id, socketio):
    try:
        emit_progress(session_id, "Creating prediction map", socketio)
        os.makedirs('output', exist_ok=True)

        emit_progress(session_id, "Plotting Nyeri boundaries", socketio)
        fig, ax = plt.subplots(figsize=(12, 8))
        nyeri_gdf.plot(ax=ax, alpha=0.5, edgecolor='k')
        
        emit_progress(session_id, "Plotting predicted suitability on grid", socketio)
        grid_gdf.plot(ax=ax, column='Predicted_Suitability', cmap='viridis', legend=True, alpha=0.7)
        
        ax.set_title('Predicted Suitability Map')
        prediction_map_path = os.path.join('output', f'predicted_suitability_map_session_{session_id}.png')
        plt.savefig(prediction_map_path)
        plt.close()
        
        emit_progress(session_id, f"Prediction map created: {prediction_map_path}", socketio)
        return prediction_map_path

    except Exception as e:
        error_message = f"Error in create_prediction_map: {str(e)}"
        emit_error(session_id, error_message, socketio)
        logging.error(error_message)
        logging.error(traceback.format_exc())
        return "Failed to create prediction map"

def create_data_completeness_map(grid_gdf, nyeri_gdf, session_id, socketio):
    try:
        emit_progress(session_id, "Creating data completeness map", socketio)
        os.makedirs('output', exist_ok=True)

        emit_progress(session_id, "Plotting Nyeri boundaries", socketio)
        fig, ax = plt.subplots(figsize=(12, 8))
        nyeri_gdf.plot(ax=ax, alpha=0.5, edgecolor='k')
        
        emit_progress(session_id, "Plotting data completeness on grid", socketio)
        grid_gdf.plot(ax=ax, column='data_completeness', cmap='RdYlGn', legend=True, alpha=0.7)
        
        ax.set_title('Data Completeness Map')
        completeness_map_path = os.path.join('output', f'data_completeness_map_session_{session_id}.png')
        plt.savefig(completeness_map_path)
        plt.close()
        
        emit_progress(session_id, f"Data completeness map created: {completeness_map_path}", socketio)
        return completeness_map_path

    except Exception as e:
        error_message = f"Error in create_data_completeness_map: {str(e)}"
        emit_error(session_id, error_message, socketio)
        logging.error(error_message)
        logging.error(traceback.format_exc())
        return "Failed to create data completeness map"

def create_variable_distribution_plots(grid_gdf, session_id, socketio):
    try:
        emit_progress(session_id, "Creating variable distribution plots", socketio)
        continuous_vars = ['Buffer_River', 'Buffer_Road', 'Buffer_ProtectedArea', 'Buffer_Settlement', 'Slope']
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, var in enumerate(continuous_vars):
            if var in grid_gdf.columns:
                sns.histplot(data=grid_gdf, x=var, ax=axes[i], kde=True)
                axes[i].set_title(f'Distribution of {var}')
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Frequency')
                logging.info(f"Created distribution plot for variable: {var}")
            else:
                logging.warning(f"Variable '{var}' not found in grid_gdf. Skipping distribution plot.")
                axes[i].set_visible(False)

        plt.tight_layout()
        dist_plot_path = os.path.join('output', f'variable_distributions_session_{session_id}.png')
        plt.savefig(dist_plot_path)
        plt.close()
        
        emit_progress(session_id, f"Variable distribution plots created: {dist_plot_path}", socketio)
        return dist_plot_path

    except Exception as e:
        error_message = f"Error in create_variable_distribution_plots: {str(e)}"
        emit_error(session_id, error_message, socketio)
        logging.error(error_message)
        logging.error(traceback.format_exc())
        return "Failed to create variable distribution plots"

def perform_sensitivity_analysis(grid_gdf, original_weights, session_id, socketio):
    try:
        emit_progress(session_id, "Starting sensitivity analysis", socketio)
        os.makedirs('output', exist_ok=True)
        sensitivity_results = {}
        weight_variations = np.arange(0.5, 1.51, 0.1)
        
        total_vars = len(original_weights)
        for idx, var in enumerate(original_weights.keys()):
            emit_progress(session_id, f"Sensitivity analysis: Processing variable '{var}' ({idx + 1}/{total_vars})", socketio)
            var_sensitivity = []
            for weight in weight_variations:
                temp_weights = original_weights.copy()
                temp_weights[var] = weight
                
                # Convert to NumPy arrays for element-wise multiplication
                values = np.array([grid_gdf[v].values for v in temp_weights.keys()])
                weights = np.array([temp_weights[v] for v in temp_weights.keys()])
                
                # Perform element-wise multiplication and sum
                temp_suitability = np.sum(values * weights[:, np.newaxis], axis=0)
                var_sensitivity.append(np.mean(temp_suitability))
            
            sensitivity_results[var] = var_sensitivity
            emit_progress(session_id, f"Sensitivity analysis completed for variable '{var}'", socketio)
        
        emit_progress(session_id, "Generating sensitivity analysis plots", socketio)
        fig = go.Figure()
        for var, sensitivities in sensitivity_results.items():
            fig.add_trace(go.Scatter(x=weight_variations, y=sensitivities, mode='lines', name=var))
        
        fig.update_layout(
            title='Sensitivity Analysis of Weights',
            xaxis_title='Weight Multiplier',
            yaxis_title='Mean Suitability Score',
            legend_title='Variables'
        )
        
        sensitivity_plot_path = os.path.join('output', f'sensitivity_analysis_session_{session_id}.html')
        fig.write_html(sensitivity_plot_path)
        emit_progress(session_id, "Sensitivity analysis plots generated", socketio)
        
        return sensitivity_results, sensitivity_plot_path

    except Exception as e:
        error_message = f"Error in perform_sensitivity_analysis: {str(e)}"
        emit_error(session_id, error_message, socketio)
        logging.error(error_message)
        logging.error(traceback.format_exc())
        return {}, "Failed to perform sensitivity analysis"

    except Exception as e:
        error_message = f"Error in perform_sensitivity_analysis: {str(e)}"
        emit_error(session_id, error_message, socketio)
        logging.error(error_message)
        logging.error(traceback.format_exc())
        return {}, "Failed to perform sensitivity analysis"

def create_interactive_prediction_map(grid_gdf, nyeri_gdf, session_id, socketio):
    try:
        emit_progress(session_id, "Creating interactive prediction map", socketio)
        os.makedirs('output', exist_ok=True)
        if grid_gdf.empty:
            emit_progress(session_id, "Grid GeoDataFrame is empty. Cannot create interactive prediction map.", socketio)
            return "Interactive prediction map not created due to empty grid data."

        emit_progress(session_id, "Converting geometries to GeoJSON format for interactive map", socketio)
        grid_gdf_json = grid_gdf.geometry.__geo_interface__
        
        centroid = nyeri_gdf.geometry.centroid
        mean_lat = centroid.y.mean()
        mean_lon = centroid.x.mean()
        emit_progress(session_id, f"Centering interactive map at latitude {mean_lat}, longitude {mean_lon}", socketio)

        emit_progress(session_id, "Generating interactive choropleth map using Plotly", socketio)
        fig = px.choropleth_mapbox(
            grid_gdf, 
            geojson=grid_gdf_json, 
            locations=grid_gdf.index, 
            color='Predicted_Suitability',
            color_continuous_scale="Viridis",
            range_color=(grid_gdf['Predicted_Suitability'].min(), grid_gdf['Predicted_Suitability'].max()),
            mapbox_style="carto-positron",
            zoom=9, 
                        center = {"lat": mean_lat, "lon": mean_lon},
            opacity=0.5,
            labels={'Predicted_Suitability':'Predicted Suitability'}
        )
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        
        prediction_map_path = os.path.join('output', f'interactive_predicted_suitability_map_session_{session_id}.html')
        fig.write_html(prediction_map_path)
        emit_progress(session_id, "Interactive prediction map created successfully", socketio)
        return prediction_map_path

    except Exception as e:
        error_message = f"Error in create_interactive_prediction_map: {str(e)}"
        emit_error(session_id, error_message, socketio)
        logging.error(error_message)
        logging.error(traceback.format_exc())
        return "Failed to create interactive prediction map"
    
def create_visualizations_final(sampled_points, model, X, session_id, socketio):
    """
    Wrapper function to create all necessary visualizations.

    Parameters:
    - sampled_points (GeoDataFrame): The sampled points for analysis.
    - model (Pipeline): The trained model pipeline.
    - X (DataFrame): Feature matrix.
    - session_id (str): The session identifier.
    - socketio (SocketIO): The SocketIO instance.

    Returns:
    - dict: A dictionary of plot names and their file paths.
    """
    emit_progress(session_id, "Starting to create visualizations", socketio)
    plot_paths = create_visualizations(sampled_points, model, X, session_id, socketio)
    emit_progress(session_id, "All visualizations created", socketio)
    return plot_paths


def grid_based_suitability_analysis(nyeri_gdf, layers, weights, session_id, socketio, cell_size=100, num_points=1000):
    try:
        emit_progress(session_id, "Starting grid-based suitability analysis", socketio)
        
        if nyeri_gdf is None or len(nyeri_gdf) == 0:
            raise ValueError("Input nyeri_gdf is None or empty")
        
        emit_progress(session_id, "Creating grid", socketio)
        grid_gdf = create_grid(nyeri_gdf, cell_size, session_id, socketio)
        emit_sample_data(session_id, "grid_creation", grid_gdf.head().to_dict('records'), socketio)
        
        if grid_gdf is None or len(grid_gdf) == 0:
            raise ValueError("Failed to create grid or grid is empty")
        
        emit_progress(session_id, "Cleaning grid geometries", socketio)
        grid_gdf = clean_geometries(grid_gdf, session_id, socketio)
        emit_progress(session_id, f"Grid geometries after cleaning: {len(grid_gdf)}", socketio)
        
        if len(grid_gdf) == 0:
            raise ValueError("No valid geometries in the grid GeoDataFrame after cleaning")
        
        emit_progress(session_id, "Extracting data for each layer", socketio)
        grid_gdf = extract_layer_data(grid_gdf, layers, session_id, socketio)
        emit_progress(session_id, "Data extraction for layers completed", socketio)

        emit_progress(session_id, "Saving extracted data to Excel", socketio)
        extracted_data_path = save_extracted_data(grid_gdf, session_id)
        emit_progress(session_id, f"Extracted data saved to: {extracted_data_path}", socketio)
        emit_sample_data(session_id, "extracted_data", grid_gdf.head().to_dict('records'), socketio)

        emit_progress(session_id, "Preprocessing and normalizing data", socketio)
        grid_gdf_processed = preprocess_data(grid_gdf, weights, session_id, socketio)
        emit_sample_data(session_id, "preprocessed_data", grid_gdf_processed.head().to_dict('records'), socketio)

        emit_progress(session_id, f"Sampling {min(num_points, len(grid_gdf_processed))} points for analysis", socketio)
        sampled_points = grid_gdf_processed.sample(n=min(num_points, len(grid_gdf_processed)))
        emit_progress(session_id, f"Sampled {len(sampled_points)} points for analysis", socketio)

        emit_progress(session_id, "Preparing data for machine learning", socketio)
        X, y = prepare_data_for_ml(sampled_points)
        emit_progress(session_id, "Prepared data for machine learning", socketio)

        emit_progress(session_id, "Training and evaluating models", socketio)
        model, mse, r2, cv_scores = train_and_evaluate_model(X, y)

        # Make predictions for all grid cells
        grid_X = prepare_data_for_ml(grid_gdf_processed, is_prediction=True)
        grid_gdf_processed['Predicted_Suitability'] = model.predict(grid_X)
        emit_progress(session_id, "Predictions made for all grid cells", socketio)

        emit_progress(session_id, "Creating analysis report", socketio)
        report = create_analysis_report(model, X, mse, r2, cv_scores, session_id, socketio)
        emit_progress(session_id, "Analysis report created", socketio)
        emit_sample_data(session_id, "analysis_report", {"report": report}, socketio)

        emit_progress(session_id, "Creating visualizations", socketio)
        plot_paths = create_visualizations_final(sampled_points, model, X, session_id, socketio)

        emit_progress(session_id, "Saving processed data to Excel", socketio)
        excel_path = save_to_excel(grid_gdf_processed, sampled_points, model, session_id)

        emit_progress(session_id, "Creating prediction map", socketio)
        prediction_map = create_prediction_map(grid_gdf_processed, nyeri_gdf, session_id, socketio)

        emit_progress(session_id, "Creating data completeness map", socketio)
        completeness_map = create_data_completeness_map(grid_gdf_processed, nyeri_gdf, session_id, socketio)

        emit_progress(session_id, "Creating variable distribution plots", socketio)
        dist_plots = create_variable_distribution_plots(grid_gdf_processed, session_id, socketio)

        emit_progress(session_id, "Performing sensitivity analysis", socketio)
        sensitivity_results, sensitivity_plot = perform_sensitivity_analysis(grid_gdf_processed, weights, session_id, socketio)

        emit_progress(session_id, "Creating interactive prediction map", socketio)
        interactive_prediction_map = create_interactive_prediction_map(grid_gdf_processed, nyeri_gdf, session_id, socketio)

        emit_progress(session_id, "Grid-based suitability analysis completed successfully", socketio)
        return (sampled_points, model, report, plot_paths, excel_path, grid_gdf_processed, 
                prediction_map, completeness_map, dist_plots, sensitivity_results, 
                sensitivity_plot, interactive_prediction_map)

    except Exception as e:
        error_message = f"Error in grid-based suitability analysis: {str(e)}"
        emit_error(session_id, error_message, socketio)
        logging.error(error_message)
        logging.error(traceback.format_exc())
        return None

# Additional routes and SocketIO event handlers can be added here

if __name__ == "__main__":
    # Example of running the Flask app with SocketIO
    # Replace '0.0.0.0' and port '5000' as needed
    socketio.run(app, host='0.0.0.0', port=5000)