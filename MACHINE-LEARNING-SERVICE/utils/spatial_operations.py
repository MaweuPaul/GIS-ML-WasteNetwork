import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from rasterio.plot import plotting_extent
from shapely.geometry import MultiPolygon, LineString, Polygon, MultiLineString
from shapely.ops import unary_union
from pyproj import CRS
import warnings
import os
import eventlet
from scipy.ndimage import generic_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text, inspect
import json
from tabulate import tabulate
import traceback

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Nyeri, Kenya is approximately in UTM zone 37S
NYERI_CRS = CRS.from_epsg(32737)

# Define consistent colors for buffers
BUFFER_COLORS = {
    'River': '#66c2a5',
    'Road': '#fc8d62',
    'ProtectedArea': '#8da0cb',
    'Settlement': '#e78ac3'
}

# Suitability mapping functions
def river_suitability_mapping(distance):
    if distance <= 200:
        return 1
    elif 200 < distance <= 500:
        return 2
    elif 500 < distance <= 1000:
        return 3
    elif 1000 < distance <= 1500:
        return 4
    else:
        return 5

def residential_area_suitability_mapping(distance):
    if distance <= 200:
        return 1
    elif 200 < distance <= 500:
        return 2
    elif 500 < distance <= 1000:
        return 3
    elif 1000 < distance <= 1500:
        return 4
    else:
        return 5

def soil_suitability_mapping(soil_type):
    soil_type = str(soil_type).lower()
    if soil_type == 'sand':
        return 1
    elif soil_type == 'loam':
        return 2
    elif soil_type == 'silt':
        return 4
    elif soil_type == 'clay':
        return 5
    else:
        return 0

def road_suitability_mapping(distance):
    if distance <= 200:
        return 1
    elif 200 < distance <= 500:
        return 3
    elif 500 < distance <= 1000:
        return 5
    elif 1000 < distance <= 1500:
        return 4
    else:
        return 2

def settlement_suitability_mapping(distance):
    if distance <= 200:
        return 1
    elif 200 < distance <= 500:
        return 2
    elif 500 < distance <= 1000:
        return 3
    elif 1000 < distance <= 1500:
        return 4
    else:
        return 5

def protected_areas_suitability_mapping(distance):
    if distance <= 200:
        return 1
    elif 200 < distance <= 500:
        return 2
    elif 500 < distance <= 1000:
        return 3
    elif 1000 < distance <= 1500:
        return 4
    else:
        return 5

def geology_suitability_mapping(geology_type):
    geology_type = str(geology_type).lower()
    if geology_type == 'ti':
        return 3  # Moderately suitable
    elif geology_type == 'qv':
        return 4  # Suitable
    elif geology_type == 'qc':
        return 2  # Less suitable
    else:
        return 1  # Least suitable or unknown

def slope_suitability_mapping(slope_degree):
    if slope_degree <= 5:
        return 5  # Highly suitable (0-5 degrees)
    elif 5 < slope_degree <= 10:
        return 4  # Suitable (5-10 degrees)
    elif 10 < slope_degree <= 15:
        return 3  # Moderately suitable (10-15 degrees)
    elif 15 < slope_degree <= 20:
        return 2  # Less suitable (15-20 degrees)
    else:
        return 1  # Not suitable (>20 degrees)

def emit_progress(session_id, message, socketio):
    """
    Emit progress updates to the frontend.
    """
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
    """
    Emit error messages to the frontend.
    """
    try:
        if socketio:
            socketio.emit('task_error', {'session_id': session_id, 'message': message}, room=session_id)
            print(f"Emitted error: {message}")
            eventlet.sleep(0)
        else:
            print("SocketIO instance not found. Error message:", message)
    except Exception as e:
        print(f"Failed to emit error message: {e}")

def fetch_data_from_postgis(query, engine, session_id, socketio):
    """
    Fetch data from PostGIS database.
    """
    try:
        emit_progress(session_id, f"Executing query: {query}", socketio)
        gdf = gpd.read_postgis(query, engine, geom_col='geom')
        emit_progress(session_id, f"Fetched {len(gdf)} geometries.", socketio)
        return gdf
    except Exception as e:
        emit_error(session_id, f"Error fetching data: {str(e)}", socketio)
        return None

def reproject_to_nyeri(gdf, session_id, socketio):
    """
    Reproject GeoDataFrame to Nyeri CRS.
    """
    try:
        emit_progress(session_id, f"Original CRS: {gdf.crs}", socketio)
        emit_progress(session_id, f"Reprojecting data to Nyeri CRS (EPSG:32737)", socketio)
        gdf_projected = gdf.to_crs(NYERI_CRS)
        emit_progress(session_id, f"Data reprojected to Nyeri CRS (EPSG:32737)", socketio)
        return gdf_projected
    except Exception as e:
        emit_error(session_id, f"Error reprojecting data: {str(e)}", socketio)
        return None

def create_arcgis_like_buffers(gdf, distances, session_id, socketio):
    """
    Create buffers similar to ArcGIS style.
    """
    buffers = []
    for distance in distances:
        try:
            emit_progress(session_id, f"Creating buffer of {distance} meters.", socketio)
            buffer = gdf.geometry.buffer(distance)
            merged_buffer = gpd.GeoDataFrame(geometry=[buffer.unary_union], crs=gdf.crs)
            buffers.append(merged_buffer)
            emit_progress(session_id, f"Merged buffers for {distance} meters.", socketio)
        except Exception as e:
            emit_error(session_id, f"Error creating buffer at {distance} meters: {str(e)}", socketio)
    return buffers

def plot_buffers(buffers, original_gdf, session_id, socketio, title='Buffer Zones'):
    """
    Plot buffer zones and save the plot.
    """
    try:
        emit_progress(session_id, f"Starting to plot buffers for {title}.", socketio)
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot original geometries
        original_gdf.plot(ax=ax, color='red', alpha=0.5)
        
        # Plot buffer zones
        for i, buffer in enumerate(buffers):
            buffer.plot(ax=ax, alpha=0.3, color=BUFFER_COLORS.get(title.split()[0], f'C{i}'))
        
        ax.set_title(title)
        ax.axis('off')
        
        # Save the plot
        os.makedirs('output', exist_ok=True)
        plot_path = os.path.join('output', f'{title.lower().replace(" ", "_")}_session_{session_id}.png')
        plt.savefig(plot_path)
        plt.close()
        
        emit_progress(session_id, f"Buffer zones plot for {title} saved as {plot_path}", socketio)
        return plot_path
    except Exception as e:
        emit_error(session_id, f"Error plotting buffer zones: {str(e)}", socketio)
        return None

def fetch_dem_data(engine, session_id, socketio):
    """
    Fetch DEM data from PostGIS and reproject to Nyeri CRS.
    """
    query = """
    SELECT id, name, elevation, "geometryType", 
           ST_AsText(geom) as geom_wkt
    FROM "DigitalElevationModel"
    """
    try:
        with engine.connect() as connection:
            dem_df = pd.read_sql(text(query), connection)
        if dem_df.empty:
            raise ValueError("DEM query returned no results.")
        emit_progress(session_id, f"Fetched {len(dem_df)} DEM polygons", socketio)
        emit_progress(session_id, f"DEM Geometry types: {dem_df['geometryType'].unique()}", socketio)
        emit_progress(session_id, f"Sample DEM geometry WKT: {dem_df['geom_wkt'].iloc[0][:100]}...", socketio)
        
        # Convert to GeoDataFrame
        dem_gdf = gpd.GeoDataFrame(
            dem_df,
            geometry=gpd.GeoSeries.from_wkt(dem_df['geom_wkt']),
            crs="EPSG:4326"  # Assuming WGS84
        )
        if 'geometry' not in dem_gdf.columns:
            raise ValueError("DEM data does not contain a 'geometry' column.")
        dem_gdf = reproject_to_nyeri(dem_gdf, session_id, socketio)
        return dem_gdf
    except Exception as e:
        emit_error(session_id, f"Error fetching DEM data: {str(e)}", socketio)
        return None

def calculate_slope(dem_raster, transform, session_id, socketio):
    """
    Calculate slope from DEM raster using the Horn method.
    """
    try:
        # Get cell size
        dx = transform[0]
        dy = -transform[4]

        # Pad the DEM to handle edges
        pad_dem = np.pad(dem_raster, pad_width=1, mode='edge')

        # Calculate gradients
        dz_dx = ((pad_dem[1:-1, 2:] + 2*pad_dem[1:-1, 1:-1] + pad_dem[1:-1, :-2]) - 
                 (pad_dem[:-2, 2:] + 2*pad_dem[:-2, 1:-1] + pad_dem[:-2, :-2])) / (8 * dx)
        dz_dy = ((pad_dem[2:, 1:-1] + 2*pad_dem[1:-1, 1:-1] + pad_dem[:-2, 1:-1]) - 
                 (pad_dem[2:, :-2] + 2*pad_dem[1:-1, :-2] + pad_dem[:-2, :-2])) / (8 * dy)

        # Calculate slope
        slope_radians = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_degrees = np.degrees(slope_radians)

        emit_progress(session_id, "Slope calculation successful.", socketio)
        return slope_degrees
    except Exception as e:
        emit_error(session_id, f"Error calculating slope: {str(e)}", socketio)
        return None

def visualize_slope(slope, transform, session_id, socketio):
    """
    Visualize the slope data and save as PNG and TIF.
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(slope, cmap='terrain', extent=plotting_extent(slope, transform))
        cbar = fig.colorbar(im, ax=ax, orientation='vertical')
        cbar.set_label('Slope (degrees)')
        ax.set_title('Slope Map')
        ax.set_axis_off()
        
        os.makedirs('output', exist_ok=True)
        slope_plot_path = os.path.join('output', f'slope_map_session_{session_id}.png')
        plt.savefig(slope_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Save as TIF
        slope_tif_path = os.path.join('output', f'slope_map_session_{session_id}.tif')
        with rasterio.open(
            slope_tif_path,
            'w',
            driver='GTiff',
            height=slope.shape[0],
            width=slope.shape[1],
            count=1,
            dtype=slope.dtype,
            crs=NYERI_CRS,
            transform=transform
        ) as dst:
            dst.write(slope, 1)
        
        emit_progress(session_id, f"Slope map saved as PNG: {slope_plot_path}", socketio)
        emit_progress(session_id, f"Slope map saved as TIF: {slope_tif_path}", socketio)
        return slope_plot_path, slope_tif_path
    except Exception as e:
        emit_error(session_id, f"Error visualizing slope: {str(e)}", socketio)
        return None, None

def fetch_and_classify_vector(table_name, classification_func, engine, session_id, socketio):
    """
    Fetch vector data from PostGIS and classify it based on the provided function.
    """
    try:
        emit_progress(session_id, f"Fetching and classifying {table_name} data.", socketio)
        
        # Construct the query based on available columns
        if table_name == 'Geology':
            query = f'SELECT id, geom, properties FROM "{table_name}"'
        elif table_name == 'Soil':
            query = f'SELECT id, geom, "soilType" FROM "{table_name}"'
        else:
            raise ValueError(f"Unsupported table: {table_name}")
        
        gdf = gpd.read_postgis(query, engine, geom_col='geom')
        
        if 'geom' not in gdf.columns:
            raise ValueError(f"Required 'geom' column not found in {table_name} table.")
        
        gdf = gdf.set_geometry('geom')
        gdf = reproject_to_nyeri(gdf, session_id, socketio)
        
        if table_name == 'Geology':
            # Extract 'GLG' from properties JSON
            gdf['classification_field'] = gdf['properties'].apply(lambda x: json.loads(x)['GLG'] if isinstance(x, str) else x.get('GLG') if isinstance(x, dict) else None)
        elif table_name == 'Soil':
            gdf['classification_field'] = gdf['soilType']
        
        gdf['suitability_score'] = gdf['classification_field'].apply(classification_func)
        emit_progress(session_id, f"Fetched and classified {table_name} data.", socketio)
        return gdf
    except Exception as e:
        emit_error(session_id, f"Error fetching and classifying {table_name} data: {str(e)}", socketio)
        return None

def fetch_and_plot_landuse_raster(engine, session_id, socketio):
    """
    Fetch land use raster from PostGIS, convert it to a numpy array, and plot it using MemoryFile.
    """
    query = """
    SELECT raster
    FROM "LandUseRaster"
    LIMIT 1
    """
    try:
        emit_progress(session_id, "Fetching land use raster.", socketio)
        with engine.connect() as connection:
            result = connection.execute(text(query)).fetchone()
        
        if result is None:
            raise ValueError("Land use query returned no results.")
        
        rast_bytes = result[0]
        
        # Use MemoryFile to read the raster bytes
        with MemoryFile(rast_bytes) as memfile:
            with memfile.open() as dataset:
                rast_array = dataset.read(1)  # Assuming single band raster
                transform = dataset.transform
                crs = dataset.crs
        
        emit_progress(session_id, f"Fetched land use raster with shape {rast_array.shape}", socketio)
        
        # Plot the raster
        plt.figure(figsize=(12, 8))
        plt.imshow(rast_array, cmap='viridis')
        plt.colorbar(label='Land Use Category')
        plt.title('Land Use Raster')
        plt.axis('off')
        
        # Save the plot
        os.makedirs('output', exist_ok=True)
        plot_path = os.path.join('output', f'landuse_raster_plot_session_{session_id}.png')
        plt.savefig(plot_path)
        plt.close()
        
        emit_progress(session_id, f"Land use raster plot saved as {plot_path}", socketio)
        
        # Analyze land use categories
        unique_values, counts = np.unique(rast_array, return_counts=True)
        total_pixels = np.sum(counts)
        emit_progress(session_id, "Land use categories and their extents:", socketio)

        land_use_types = {0: 'No Data', 1: 'Forests', 2: 'Bareland', 3: 'Buildup', 4: 'Farmland'}
        category_info = []  # Initialize category_info list
        for value, count in zip(unique_values, counts):
            percentage = (count / total_pixels) * 100
            land_use_type = land_use_types.get(value, f'Unknown ({value})')
            category_info.append({
                'value': int(value),
                'type': land_use_type,
                'count': int(count),
                'percentage': float(percentage)
            })
            emit_progress(session_id, f"Category {value} ({land_use_type}): {count} pixels, {percentage:.2f}% of total area", socketio)
        
        # Emit category information to frontend
        socketio.emit('landuse_categories', {
            'session_id': session_id,
            'categories': category_info,
            'plot_path': plot_path
        }, room=session_id)
        
        return rast_array, transform, crs, plot_path, land_use_types
    except Exception as e:
        emit_error(session_id, f"Error fetching and plotting land use raster: {str(e)}", socketio)
        return None, None, None, None, None

def process_landuse_suitability(landuse_raster, land_use_types, session_id, socketio):
    """
    Process land use suitability based on the land use raster.
    """
    try:
        def landuse_suitability_mapping(land_use):
            land_use_type = land_use_types.get(land_use, '').lower()
            if 'forests' in land_use_type:
                return 1  # Least suitable
            elif 'bareland' in land_use_type:
                return 5  # Most suitable
            elif 'buildup' in land_use_type:
                return 2  # Less suitable
            elif 'farmland' in land_use_type:
                return 4  # Suitable
            else:
                return 0  # Unknown land use type or No Data
        
        landuse_suitability = np.vectorize(landuse_suitability_mapping)(landuse_raster)
        emit_progress(session_id, "Land use suitability mapping applied.", socketio)
        return landuse_suitability
    except Exception as e:
        emit_error(session_id, f"Error processing land use suitability: {str(e)}", socketio)
        return None

def train_suitability_model(X, y, session_id, socketio):
    """
    Train a Random Forest model for suitability prediction.
    """
    try:
        emit_progress(session_id, "Starting ML model training.", socketio)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred)
        
        emit_progress(session_id, f"ML model training completed. Classification report:\n{report}", socketio)
        return model, scaler
    except Exception as e:
        emit_error(session_id, f"Error training ML model: {str(e)}", socketio)
        return None, None

def apply_ml_model(total_suitability, model, scaler, rows, cols, transform, session_id, socketio):
    """
    Apply the trained ML model to predict suitability.
    """
    try:
        emit_progress(session_id, "Applying ML model for suitability prediction.", socketio)
        X = total_suitability.reshape(-1, 1)
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        ml_suitability = y_pred.reshape(rows, cols)
        
        # Visualize and save the ML suitability map
        plt.figure(figsize=(12, 8))
        plt.imshow(ml_suitability, cmap='viridis')
        plt.colorbar(label='ML Predicted Suitability')
        plt.title('ML Predicted Suitability Map')
        plt.axis('off')
        ml_suitability_map_path = os.path.join('output', f'ml_suitability_map_session_{session_id}.png')
        plt.savefig(ml_suitability_map_path)
        plt.close()
        
        emit_progress(session_id, f"ML suitability map saved as {ml_suitability_map_path}", socketio)
        
        # Save the ML suitability raster
        ml_suitability_raster_path = os.path.join('output', f'ml_suitability_map_session_{session_id}.tif')
        with rasterio.open(
            ml_suitability_raster_path,
            'w',
            driver='GTiff',
            height=rows,
            width=cols,
            count=1,
            dtype='uint8',
            crs=NYERI_CRS,
            transform=transform
        ) as dst:
            dst.write(ml_suitability.astype('uint8'), 1)
        
        emit_progress(session_id, f"ML suitability raster saved as {ml_suitability_raster_path}", socketio)
        return ml_suitability_map_path, ml_suitability_raster_path
    except Exception as e:
        emit_error(session_id, f"Error applying ML model: {str(e)}", socketio)
        return None, None

def calculate_weights():
    """
    Calculate weights for landfill site suitability criteria using AHP with whole numbers.
    """
    # Define the criteria based on the provided datasets
    criteria = [
        'River',
        'Residential Area',
        'Soil',
        'Road',
        'Settlement',
        'Protected Areas',
        'Geology',
        'Land Use'
    ]
    n = len(criteria)
    
    # Whole number-based pairwise comparison matrix following Saaty's scale (1, 3, 5, 7, 9)
    matrix = np.array([
        [1,      3,          5,      5,         3,              3,            5,        5],  # River
        [1/3,      1,          3,      3,         3,              3,            3,        3],  # Residential Area
        [1/5, 1/3,          1,      1,         1,              1,            1,        1],  # Soil
        [1/5, 1/3,          1,      1,         1,              1,            1,        1],  # Road
        [1/3, 1/3,          1,      1,         1,              1,            1,        1],  # Settlement
        [1/3, 1/3,          1,      1,         1,              1,            1,        1],  # Protected Areas
        [1/5, 1/3,          1,      1,         1,              1,            1,        1],  # Geology
        [1/5, 1/3,          1,      1,         1,              1,            1,        1],  # Land Use
    ])

    # Calculate weights using the eigenvector method
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_index = np.argmax(eigenvalues.real)
    weights = eigenvectors[:, max_index].real
    weights = weights / np.sum(weights)

    # Convert weights to percentages
    weights_percent = weights * 100

    # Create a dictionary of criteria and their corresponding weights
    weights_dict = dict(zip(criteria, weights_percent))

    return weights_dict, matrix

def calculate_consistency_ratio(matrix, weights):
    """
    Calculate the Consistency Ratio (CR) to assess the consistency of the pairwise comparisons.
    """
    n = matrix.shape[0]
    # Calculate lambda_max
    weighted_sum = np.dot(matrix, weights)
    lambda_max = np.sum(weighted_sum / weights) / n

    # Consistency Index (CI)
    consistency_index = (lambda_max - n) / (n - 1)

    # Random Index (RI) values for different n
    random_index = {3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24,
                   7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    ri = random_index.get(n, 1.49)  # Default to 1.49 if n > 10

    # Consistency Ratio (CR)
    consistency_ratio = consistency_index / ri

    return consistency_ratio

def run_full_spatial_operations(engine, session_id, socketio):
    try:
        emit_progress(session_id, "Starting full spatial operations.", socketio)
        
        # Use predefined weights
        weights_dict, matrix = calculate_weights()
        emit_progress(session_id, f"Using predefined AHP Weights: {weights_dict}", socketio)
        
        # Calculate and report consistency ratio
        normalized_weights = np.array([weight / 100 for weight in weights_dict.values()])
        consistency_ratio = calculate_consistency_ratio(matrix, normalized_weights)
        emit_progress(session_id, f"Consistency Ratio: {consistency_ratio:.4f}", socketio)
        if consistency_ratio < 0.1:
            emit_progress(session_id, "The pairwise comparison matrix is consistent (CR < 0.1).", socketio)
        else:
            emit_progress(session_id, "Warning: The pairwise comparison matrix is not consistent (CR >= 0.1).", socketio)
            emit_progress(session_id, "Please revise the comparison matrix to improve consistency.", socketio)
        
        # Define feature types and their respective buffer distances
        feature_types = [
            ('River', [200, 500, 1000, 1500]),
            ('Road', [200, 500, 1000]),
            ('ProtectedArea', [200, 500, 1000, 1500]),
            ('Settlement', [200, 500, 1000, 1500])
        ]
        
        all_buffers = []
        all_original = []
        buffer_images = {}
        
        for feature_type, distances in feature_types:
            emit_progress(session_id, f"Processing {feature_type} data.", socketio)
            
            # Fetch data from PostGIS
            query = f'SELECT id, geom FROM "{feature_type}"'
            gdf = fetch_data_from_postgis(query, engine, session_id, socketio)
            
            if gdf is not None:
                # Reproject to Nyeri CRS
                gdf_projected = reproject_to_nyeri(gdf, session_id, socketio)
                
                # Ensure the geometry column is named 'geometry'
                if 'geom' in gdf_projected.columns:
                    gdf_projected = gdf_projected.rename(columns={'geom': 'geometry'}).set_geometry('geometry')
                
                # Get unique geometry types
                geometry_types = gdf_projected.geometry.geom_type.unique()
                emit_progress(session_id, f"Geometry types for {feature_type}: {geometry_types}", socketio)
                
                # Create buffers
                buffers = create_arcgis_like_buffers(gdf_projected, distances, session_id, socketio)
                
                if buffers:
                    all_buffers.extend(buffers)
                    all_original.append(gdf_projected)
                    
                    # Plot buffers
                    buffer_plot_path = plot_buffers(buffers, gdf_projected, session_id, socketio, title=f'{feature_type} Buffers')
                    if buffer_plot_path:
                        buffer_images[feature_type] = buffer_plot_path
        
        # Create combined buffer zones plot
        if all_buffers and all_original:
            emit_progress(session_id, "Creating combined buffer zones plot.", socketio)
            try:
                # Concatenate all original GeoDataFrames
                concatenated_original = pd.concat(all_original, ignore_index=True)
                
                # Debug: Print columns and data types
                emit_progress(session_id, f"Columns in concatenated GeoDataFrame: {concatenated_original.columns}", socketio)
                emit_progress(session_id, f"Data types: {concatenated_original.dtypes}", socketio)
                
                # Ensure 'geometry' column is present and set as active geometry column
                if 'geometry' in concatenated_original.columns:
                    concatenated_original = gpd.GeoDataFrame(concatenated_original, geometry='geometry', crs=NYERI_CRS)
                else:
                    raise ValueError("No 'geometry' column found in concatenated data.")
                
                combined_image_path = plot_buffers(
                    all_buffers, 
                    concatenated_original, 
                    session_id, 
                    socketio, 
                    title='Combined Buffer Zones'
                )
                if combined_image_path:
                    buffer_images['Combined'] = combined_image_path
            except Exception as e:
                emit_error(session_id, f"Error creating combined buffer zones: {str(e)}", socketio)
        else:
            emit_error(session_id, "No valid data to create combined buffer zones.", socketio)
        
         # DEM and slope
        emit_progress(session_id, "Starting DEM and slope analysis.", socketio)
        dem_gdf = fetch_dem_data(engine, session_id, socketio)
        if dem_gdf is not None:
            # Rasterize DEM
            bounds = dem_gdf.total_bounds
            res = 30  # 30m resolution
            rows = int((bounds[3] - bounds[1]) / res)
            cols = int((bounds[2] - bounds[0]) / res)
            transform = from_bounds(*bounds, cols, rows)
            
            dem_raster = rasterize(
                [(geom, value) for geom, value in zip(dem_gdf.geometry, dem_gdf.elevation)],
                out_shape=(rows, cols),
                transform=transform,
                fill=np.nan,
                all_touched=True ,
                dtype='float32'
            )
            
            # Handle NaN values by filling them
            from scipy.ndimage import generic_filter

            def fill_nan(array):
                # Replace NaN with the mean of the neighborhood
                if np.isnan(array).all():
                    return np.nan
                return np.nanmean(array)
            
            dem_raster_filled = generic_filter(dem_raster, fill_nan, size=3, mode='nearest')
            emit_progress(session_id, "NaN values in DEM raster have been filled.", socketio)
            
            # Calculate slope
            def calculate_slope(dem, transform):
                dx, dy = np.gradient(dem)
                cellsize_x = transform.a
                cellsize_y = -transform.e
                
                slope_rad = np.arctan(np.sqrt((dx / cellsize_x)**2 + (dy / cellsize_y)**2))
                slope_deg = np.degrees(slope_rad)
                
                return slope_deg

            # Ensure DEM raster has sufficient size for gradient calculation
            if dem_raster_filled.shape[0] < 2 or dem_raster_filled.shape[1] < 2:
                raise ValueError("DEM raster is too small for gradient calculation.")
            
            slope = calculate_slope(dem_raster_filled, transform)
            emit_progress(session_id, "Slope calculation successful.", socketio)
            
            # Visualize and save the slope
            plt.figure(figsize=(15, 15))
            extent = plotting_extent(dem_raster_filled, transform)
            plt.imshow(slope, cmap='terrain', vmin=0, vmax=45, extent=extent)
            plt.title('Slope Map', fontsize=16)
            plt.colorbar(label='Slope (degrees)')
            plt.axis('off')
            
            slope_plot_path = os.path.join('output', f'slope_map_session_{session_id}.png')
            plt.savefig(slope_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            buffer_images['Slope'] = slope_plot_path
            
            # Save the slope raster
            slope_tif_path = os.path.join('output', f'slope_map_session_{session_id}.tif')
            with rasterio.open(
                slope_tif_path,
                'w',
                driver='GTiff',
                height=slope.shape[0],
                width=slope.shape[1],
                count=1,
                dtype='float32',
                crs=NYERI_CRS,
                transform=transform
            ) as dst:
                dst.write(slope, 1)
            
            emit_progress(session_id, f"Slope analysis completed and results saved: {slope_tif_path}", socketio)

        # Geology
        geology_gdf = fetch_and_classify_vector('Geology', geology_suitability_mapping, engine, session_id, socketio)

        # Soil
        soil_gdf = fetch_and_classify_vector('Soil', soil_suitability_mapping, engine, session_id, socketio)

        # Land Use
        emit_progress(session_id, "Starting land use raster processing", socketio)
        landuse_raster, landuse_transform, landuse_crs, landuse_plot_path, land_use_types = fetch_and_plot_landuse_raster(engine, session_id, socketio)
        if landuse_raster is not None:
            buffer_images['LandUse'] = landuse_plot_path
            landuse_suitability = process_landuse_suitability(landuse_raster, land_use_types, session_id, socketio)
            emit_progress(session_id, "Land use suitability processing completed", socketio)
        else:
            emit_error(session_id, "Failed to fetch land use raster.", socketio)
            landuse_suitability = None

        # Create a common grid for all layers
        emit_progress(session_id, "Starting to create common grid for all layers", socketio)
        if all_original:
            emit_progress(session_id, "all_original is not empty, proceeding with grid creation", socketio)
            try:
                bounds = concatenated_original.total_bounds
                emit_progress(session_id, f"Calculated bounds: {bounds}", socketio)
            except Exception as e:
                emit_error(session_id, f"Error calculating bounds: {str(e)}", socketio)
                raise

            res = 30  # 30m resolution, adjust as needed
            rows = int((bounds[3] - bounds[1]) / res)
            cols = int((bounds[2] - bounds[0]) / res)
            transform = from_bounds(*bounds, cols, rows)
            emit_progress(session_id, f"Created transform with rows={rows}, cols={cols}", socketio)

            # Rasterize all vector layers
            layers = {}
            emit_progress(session_id, "Starting to rasterize vector layers", socketio)
            for i, (feature_type, _) in enumerate(feature_types):
                try:
                    layers[f'Buffer_{feature_type}'] = rasterize(
                        [(geom, 1) for geom in all_buffers[i].geometry],
                        out_shape=(rows, cols),
                        transform=transform,
                        fill=0,
                        dtype='uint8'
                    )
                    emit_progress(session_id, f"Rasterized {feature_type} buffer", socketio)
                except Exception as e:
                    emit_error(session_id, f"Error rasterizing {feature_type} buffer: {str(e)}", socketio)

            if geology_gdf is not None:
                try:
                    layers['Geology'] = rasterize(
                        [(geom, value) for geom, value in zip(geology_gdf.geometry, geology_gdf.suitability_score)],
                        out_shape=(rows, cols),
                        transform=transform,
                        fill=0,
                        dtype='float32'
                    )
                    emit_progress(session_id, "Rasterized Geology layer", socketio)
                except Exception as e:
                    emit_error(session_id, f"Error rasterizing Geology layer: {str(e)}", socketio)

            if soil_gdf is not None:
                try:
                    layers['Soil'] = rasterize(
                        [(geom, value) for geom, value in zip(soil_gdf.geometry, soil_gdf.suitability_score)],
                        out_shape=(rows, cols),
                        transform=transform,
                        fill=0,
                        dtype='float32'
                    )
                    emit_progress(session_id, "Rasterized Soil layer", socketio)
                except Exception as e:
                    emit_error(session_id, f"Error rasterizing Soil layer: {str(e)}", socketio)

            if landuse_suitability is not None:
                try:
                    # Resample land use suitability to match the common grid
                    resampled_landuse = np.zeros((rows, cols), dtype='float32')
                    rasterio.warp.reproject(
                        source=landuse_suitability,
                        destination=resampled_landuse,
                        src_transform=landuse_transform,
                        src_crs=landuse_crs,
                        dst_transform=transform,
                        dst_crs=NYERI_CRS,
                        resampling=rasterio.warp.Resampling.nearest
                    )
                    layers['LandUse'] = resampled_landuse
                    emit_progress(session_id, "Resampled Land Use layer", socketio)
                except Exception as e:
                    emit_error(session_id, f"Error resampling Land Use layer: {str(e)}", socketio)

            # Add slope suitability to layers if available
            if slope is not None:
                try:
                    resampled_slope = np.zeros((rows, cols), dtype='float32')
                    rasterio.warp.reproject(
                        source=slope,
                        destination=resampled_slope,
                        src_transform=transform,
                        src_crs=NYERI_CRS,
                        dst_transform=transform,
                        dst_crs=NYERI_CRS,
                        resampling=rasterio.warp.Resampling.bilinear
                    )
                    layers['Slope'] = np.vectorize(slope_suitability_mapping)(resampled_slope)
                    emit_progress(session_id, "Added Slope layer", socketio)
                except Exception as e:
                    emit_error(session_id, f"Error adding Slope layer: {str(e)}", socketio)

            # Calculate weighted suitability
            emit_progress(session_id, "Starting weighted suitability calculation", socketio)
            total_suitability = np.zeros((rows, cols))
            for layer_name, layer in layers.items():
                if layer_name in weights_dict:
                    total_suitability += layer * weights_dict[layer_name]
                elif layer_name.startswith('Buffer_'):
                    total_suitability += layer * (weights_dict.get('Buffers', 1) / len([l for l in layers.keys() if l.startswith('Buffer_')]))
                else:
                    total_suitability += layer * weights_dict.get(layer_name, 0)
                emit_progress(session_id, f"Added {layer_name} to total suitability", socketio)

            # Normalize total suitability to 0-100 range
            with np.errstate(invalid='ignore', divide='ignore'):
                total_suitability = ((total_suitability - np.nanmin(total_suitability)) / 
                                     (np.nanmax(total_suitability) - np.nanmin(total_suitability))) * 100
            total_suitability = np.nan_to_num(total_suitability)  # Replace NaN with 0

            # Visualize and save the total suitability map
            plt.figure(figsize=(12, 8))
            plt.imshow(total_suitability, cmap='viridis')
            plt.colorbar(label='Suitability Score')
            plt.title('Total Weighted Suitability')
            plt.axis('off')
            suitability_map_path = os.path.join('output', f'total_suitability_map_session_{session_id}.png')
            plt.savefig(suitability_map_path)
            plt.close()

            emit_progress(session_id, f"Total suitability map saved as {suitability_map_path}", socketio)

            # Save the suitability raster
            suitability_raster_path = os.path.join('output', f'total_suitability_map_session_{session_id}.tif')
            with rasterio.open(
                suitability_raster_path,
                'w',
                driver='GTiff',
                height=rows,
                width=cols,
                count=1,
                dtype='float32',
                crs=NYERI_CRS,
                transform=transform
            ) as dst:
                dst.write(total_suitability.astype('float32'), 1)

            emit_progress(session_id, f"Total suitability raster saved as {suitability_raster_path}", socketio)

            # Prepare data for ML model
            X = total_suitability.reshape(-1, 1)
            y = (total_suitability > np.percentile(total_suitability, 75)).astype(int).ravel()
            
            # Train ML model
            model, scaler = train_suitability_model(X, y, session_id, socketio)
            
            if model is not None and scaler is not None:
                # Apply ML model
                ml_suitability_map_path, ml_suitability_raster_path = apply_ml_model(
                    total_suitability, model, scaler, rows, cols, transform, session_id, socketio
                )
                
                if ml_suitability_map_path:
                    buffer_images['MLSuitability'] = ml_suitability_map_path

            # Emit buffer image paths and suitability map paths to frontend
            socketio.emit('buffer_images', {'session_id': session_id, 'images': buffer_images}, room=session_id)

            emit_progress(session_id, "All spatial operations and ML predictions completed successfully.", socketio)
            socketio.emit('operation_completed', {'session_id': session_id, 'message': 'Operations completed successfully.'}, room=session_id)
        else:
            emit_error(session_id, "No valid data to create suitability map.", socketio)

    except Exception as e:
        emit_error(session_id, f"Unexpected error during spatial operations: {str(e)}", socketio)
        emit_error(session_id, f"Traceback: {traceback.format_exc()}", socketio)
    finally:
        # Clean up temporary files if needed
        pass

if __name__ == "__main__":
    # This block is for testing purposes
    from sqlalchemy import create_engine
    
    # Replace with your actual database connection string
    engine = create_engine('postgresql://username:password@host:port/database')
    
    class MockSocketIO:
        def emit(self, event, data, room=None):
            print(f"Emitted {event}: {data}")
    
    mock_socketio = MockSocketIO()
    run_full_spatial_operations(engine, "test_session", mock_socketio)