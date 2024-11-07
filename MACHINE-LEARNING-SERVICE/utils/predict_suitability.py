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
import joblib
from scipy.ndimage import gaussian_filter

def emit_progress(session_id, message, socketio):
    try:
        if socketio:
            socketio.emit('progress_update', {'session_id': session_id, 'message': message}, room=session_id)
            print(f"Emitted progress: {message}")
            eventlet.sleep(0)
        else:
            print("Progress message:", message)
    except Exception as e:
        print(f"Failed to emit progress message: {e}")

def emit_error(session_id, message, socketio):
    try:
        if socketio:
            socketio.emit('task_error', {'session_id': session_id, 'message': message}, room=session_id)
            print(f"Emitted error: {message}")
            eventlet.sleep(0)
        else:
            print("Error message:", message)
    except Exception as e:
        print(f"Failed to emit error message: {e}")

def classify_suitability(score):
    if score <= 2.0:
        return 'Not Suitable'
    elif score <= 2.75:
        return 'Less Suitable'
    elif score <= 3.5:
        return 'Moderately Suitable'
    elif score <= 4.25:
        return 'Suitable'
    else:
        return 'Highly Suitable'

def clean_data(points_gdf):
    """Clean the input data with appropriate interpolation and handling of no-data values"""
    
    # Create a copy to avoid modifying the original dataframe
    points_gdf = points_gdf.copy()
    
    # First check if all required columns exist, if not create them
    required_columns = ['River_b', 'Road_b', 'Protect', 'Settlem', 
                       'Slope', 'Land_U', 'Soil']
    
    for col in required_columns:
        if col not in points_gdf.columns:
            points_gdf[col] = np.nan
    
    # Convert columns to numeric, keeping NaN values
    for col in required_columns:
        points_gdf[col] = pd.to_numeric(points_gdf[col], errors='coerce')
    
    # Handle Land_U specifically - replace NaN with 0 (no data)
    if 'Land_U' in points_gdf.columns:
        points_gdf['Land_U'] = points_gdf['Land_U'].fillna(0)
    else:
        points_gdf['Land_U'] = 0
    
    # For other columns, use interpolation based on nearest neighbors
    columns_to_interpolate = ['River_b', 'Road_b', 'Protect', 'Settlem', 
                            'Slope', 'Soil']
    
    for col in columns_to_interpolate:
        if col in points_gdf.columns:
            # Calculate column mean for non-null values
            col_mean = points_gdf[col].mean()
            if pd.isna(col_mean):  # If all values are NaN
                col_mean = 3  # Default middle value
            
            # First try to interpolate based on nearby values
            points_gdf[col] = points_gdf[col].interpolate(method='nearest', limit_direction='both')
            
            # If any NaN values remain after interpolation, fill with column mean
            points_gdf[col] = points_gdf[col].fillna(col_mean)
            
            # Round the interpolated values to maintain the same scale as original data
            points_gdf[col] = points_gdf[col].round(0)
            
            # Ensure values stay within valid ranges
            if col == 'Protect':
                points_gdf[col] = points_gdf[col].clip(1, 5)
            elif col in ['River_b', 'Road_b', 'Settlem', 'Soil']:
                points_gdf[col] = points_gdf[col].clip(1, 5)
            elif col == 'Slope':
                points_gdf[col] = points_gdf[col].clip(0, 4)
    
    return points_gdf
def create_suitability_map(gdf, nyeri_gdf, title, output_path):
    """Create and save a suitability map with 5 classes"""
    color_dict = {
        'Not Suitable': '#d7191c',      # Dark red
        'Less Suitable': '#fdae61',     # Orange
        'Moderately Suitable': '#ffffbf',# Light yellow
        'Suitable': '#a6d96a',          # Light green
        'Highly Suitable': '#1a9641'    # Dark green
    }
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plot base map
    nyeri_gdf.plot(ax=ax, alpha=0.3, color='lightgray')
    
    # Plot suitability classes
    for suitability_class, color in color_dict.items():
        mask = gdf['Suitability_Class'] == suitability_class
        class_gdf = gdf[mask]
        if not class_gdf.empty:
            class_gdf.plot(
                ax=ax,
                color=color,
                label=suitability_class,
                alpha=0.7
            )
    
    # Customize the legend
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles, 
        labels,
        title='Suitability Classes',
        title_fontsize=12,
        fontsize=10,
        loc='upper right',
        frameon=True,
        facecolor='white',
        edgecolor='black'
    )
    
    ax.set_title(title, fontsize=16, pad=20)
    
    # Add north arrow
    ax.annotate('N', xy=(0.98, 0.98), xycoords='axes fraction',
               horizontalalignment='center', verticalalignment='center',
               fontsize=18, fontweight='bold',
               path_effects=[pe.withStroke(linewidth=3, foreground="w")])
    
    # Add scale bar
    scalebar = ScaleBar(1, 'km', dimension='si-length', location='lower right')
    ax.add_artist(scalebar)
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_candidate_map(gdf, nyeri_gdf, title, output_path):
    """Create and save a candidate sites map"""
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plot base map
    nyeri_gdf.plot(ax=ax, alpha=0.3, color='lightgray')
    
    # Plot suitable areas
    gdf.plot(
        ax=ax,
        color='#90EE90',  # Light green
        label='Suitable',
        alpha=0.7
    )
    
    # Customize the legend
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles, 
        labels,
        title='Suitability Classes',
        title_fontsize=12,
        fontsize=10,
        loc='upper right',
        frameon=True,
        facecolor='white',
        edgecolor='black'
    )
    
    ax.set_title(title, fontsize=16, pad=20)
    
    # Add north arrow
    ax.annotate('N', xy=(0.98, 0.98), xycoords='axes fraction',
               horizontalalignment='center', verticalalignment='center',
               fontsize=18, fontweight='bold',
               path_effects=[pe.withStroke(linewidth=3, foreground="w")])
    
    # Add scale bar
    scalebar = ScaleBar(1, 'km', dimension='si-length', location='lower right')
    ax.add_artist(scalebar)
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def predict_map_suitability(nyeri_gdf, buffer_sets, raster_criteria, model_path, scaler_path, interval=10, session_id=None, socketio=None):
    try:
        emit_progress(session_id, "🚀 Starting suitability prediction...", socketio)
        
        # Load model and scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Define grid points
        bounds = nyeri_gdf.total_bounds
        x_coords = np.arange(bounds[0], bounds[2], interval)
        y_coords = np.arange(bounds[1], bounds[3], interval)
        study_area_polygon = nyeri_gdf.unary_union
        
        points = []
        point_data = []
        progress_count = 0
        progress_interval = max(1, len(x_coords) * len(y_coords) // 20)
        
        emit_progress(session_id, "\n📍 Generating grid points...", socketio)
        
        # Generate points
        for x in x_coords:
            for y in y_coords:
                point = Point(x, y)
                if study_area_polygon.contains(point):
                    point_info = {
                        'geometry': point,
                        'x': x,
                        'y': y
                    }
                    
                    # Process buffer sets
                    for feature_name, buffer_list in buffer_sets.items():
                        if not isinstance(buffer_list, list):
                            continue
                        
                        for i, buffer_gdf in enumerate(buffer_list):
                            if any(buffer_gdf.geometry.contains(point)):
                                point_info[f'{feature_name}_buffer'] = i + 1
                                break
                        else:
                            point_info[f'{feature_name}_buffer'] = len(buffer_list) + 1
                    
                    points.append(point)
                    point_data.append(point_info)
                
                progress_count += 1
                if progress_count % progress_interval == 0:
                    progress_percent = (progress_count / (len(x_coords) * len(y_coords))) * 100
                    emit_progress(session_id, f"⏳ Processed {progress_count} points ({progress_percent:.1f}%)", socketio)
        
        points_gdf = gpd.GeoDataFrame(point_data, geometry='geometry', crs=nyeri_gdf.crs)
        
        # Sample raster values
        emit_progress(session_id, "\n🔍 Sampling raster criteria values...", socketio)
        for raster_name, raster_path in raster_criteria.items():
            try:
                with rasterio.open(raster_path) as src:
                    coords = [(p.x, p.y) for p in points_gdf.geometry]
                    values = [val[0] for val in src.sample(coords)]
                    points_gdf[raster_name] = values
            except Exception as e:
                emit_error(session_id, f"⚠️ Error processing {raster_name}: {str(e)}", socketio)
                # Instead of failing, set default values
                points_gdf[raster_name] = 3  # Default middle value
        
        # Rename columns
        column_mapping = {
            'River_buffer': 'River_b',
            'Road_buffer': 'Road_b',
            'Protected_Areas_buffer': 'Protect',
            'Settlement_buffer': 'Settlem',
            'landuse': 'Land_U',
            'land_use': 'Land_U',
            'slope': 'Slope'
        }
        points_gdf = points_gdf.rename(columns=column_mapping)
        
        # Clean and interpolate data
        emit_progress(session_id, "\n🧹 Cleaning and interpolating data...", socketio)
        points_gdf = clean_data(points_gdf)
        
        # Make predictions
        emit_progress(session_id, "\n🤖 Making predictions...", socketio)
        feature_columns = ['River_b', 'Road_b', 'Settlem', 'Soil', 'Protect', 'Land_U', 'Slope']
        X = points_gdf[feature_columns].values
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        points_gdf['Total_Suit'] = predictions.round(2)
        
        # Classify suitability
        points_gdf['Suitability_Class'] = points_gdf['Total_Suit'].apply(classify_suitability)
        
        # Create output directory
        output_dir = os.path.join(os.getcwd(), 'output')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create full suitability map
        emit_progress(session_id, "\n🎨 Creating full suitability map...", socketio)
        full_map_path = os.path.join(output_dir, f"suitability_map_{session_id}_{timestamp}.png")
        create_suitability_map(
            points_gdf,
            nyeri_gdf,
            'Landfill Suitability Map',
            full_map_path
        )
        
        # Create candidate sites map
        emit_progress(session_id, "\n🎯 Creating candidate sites map...", socketio)
        candidate_gdf = points_gdf[points_gdf['Total_Suit'] >= 3.5].copy()
        candidate_map_path = os.path.join(output_dir, f"candidate_sites_map_{session_id}_{timestamp}.png")
        create_candidate_map(
            candidate_gdf,
            nyeri_gdf,
            'Candidate Landfill Sites\n(Suitability ≥ 3.5)',
            candidate_map_path
        )
        
        # Save to CSV
        full_csv_path = os.path.join(output_dir, f"full_suitability_{session_id}_{timestamp}.csv")
        candidate_csv_path = os.path.join(output_dir, f"candidate_sites_{session_id}_{timestamp}.csv")
        points_gdf.to_csv(full_csv_path, index=False)
        candidate_gdf.to_csv(candidate_csv_path, index=False)
        
        # Calculate statistics
        stats = {
            'full_map': {
                'total_points': len(points_gdf),
                'min_score': points_gdf['Total_Suit'].min(),
                'max_score': points_gdf['Total_Suit'].max(),
                'mean_score': points_gdf['Total_Suit'].mean(),
                'class_distribution': points_gdf['Suitability_Class'].value_counts().to_dict()
            },
            'candidate_sites': {
                'total_points': len(candidate_gdf),
                'min_score': candidate_gdf['Total_Suit'].min(),
                'max_score': candidate_gdf['Total_Suit'].max(),
                'mean_score': candidate_gdf['Total_Suit'].mean(),
                'class_distribution': candidate_gdf['Suitability_Class'].value_counts().to_dict()
            }
        }
        
        # Emit statistics
        emit_progress(session_id, "\n📊 Analysis Results:", socketio)
        emit_progress(session_id, f"\nFull Map Statistics:", socketio)
        for class_name, count in stats['full_map']['class_distribution'].items():
            percentage = (count / stats['full_map']['total_points']) * 100
            emit_progress(session_id, f"  • {class_name}: {count} points ({percentage:.1f}%)", socketio)
        
        emit_progress(session_id, f"\nCandidate Sites Statistics:", socketio)
        for class_name, count in stats['candidate_sites']['class_distribution'].items():
            percentage = (count / stats['candidate_sites']['total_points']) * 100
            emit_progress(session_id, f"  • {class_name}: {count} points ({percentage:.1f}%)", socketio)
        
        emit_progress(session_id, f"\n✅ Analysis complete!", socketio)
        emit_progress(session_id, f"Full suitability map saved to: {full_map_path}", socketio)
        emit_progress(session_id, f"Candidate sites map saved to: {candidate_map_path}", socketio)
        
        return {
            'full_map_path': full_map_path,
            'candidate_map_path': candidate_map_path,
            'full_csv_path': full_csv_path,
            'candidate_csv_path': candidate_csv_path,
            'stats': stats
        }

    except Exception as e:
        emit_error(session_id, f"Error in predict_suitability: {str(e)}", socketio)
        traceback.print_exc()
        return None