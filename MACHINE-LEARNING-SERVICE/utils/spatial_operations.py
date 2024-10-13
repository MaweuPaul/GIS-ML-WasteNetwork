from flask_socketio import SocketIO, emit
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from rasterio.plot import plotting_extent 
from shapely import wkt
from shapely.geometry import MultiPolygon, LineString, Polygon, MultiLineString
from shapely.ops import unary_union
from pyproj import CRS
import warnings
import os
import eventlet
from scipy.ndimage import generic_filter

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
    Fetch data from PostGIS using the provided SQL query.
    """
    emit_progress(session_id, f"Executing query: {query}", socketio)
    try:
        gdf = gpd.read_postgis(query, engine, geom_col='geom')
        if 'geom' not in gdf.columns:
            raise ValueError("Query result does not contain a 'geom' column.")
        if gdf.empty:
            raise ValueError("Query returned no results.")
        emit_progress(session_id, f"Fetched {len(gdf)} geometries.", socketio)
        emit_progress(session_id, f"Original CRS: {gdf.crs}", socketio)
        return gdf
    except Exception as e:
        emit_error(session_id, f"Error fetching data from PostGIS: {str(e)}", socketio)
        return None

def reproject_to_nyeri(gdf, session_id, socketio):
    """
    Reproject the GeoDataFrame to Nyeri CRS (EPSG:32737).
    """
    try:
        emit_progress(session_id, f"Reprojecting data to Nyeri CRS (EPSG:32737)", socketio)
        gdf_projected = gdf.to_crs(NYERI_CRS)
        emit_progress(session_id, f"Data reprojected to Nyeri CRS (EPSG:32737)", socketio)
        return gdf_projected
    except Exception as e:
        emit_error(session_id, f"Error reprojecting data to Nyeri CRS: {str(e)}", socketio)
        return gdf

def create_arcgis_like_buffers(gdf, distances, session_id, socketio):
    """
    Create buffers around the geometries in the GeoDataFrame for each specified distance.
    """
    try:
        buffered_gdfs = []
        for distance in distances:
            emit_progress(session_id, f"Creating buffer of {distance} meters.", socketio)
            buffered_geoms = []
            for geom in gdf.geometry:
                try:
                    if isinstance(geom, MultiLineString):
                        buffered = MultiPolygon([LineString(part).buffer(distance, cap_style=3, join_style=2, resolution=32) 
                                                 for part in geom.geoms])
                    else:
                        buffered = geom.buffer(distance, cap_style=3, join_style=2, resolution=32)
                    
                    # Ensure the buffered geometry is valid
                    if not buffered.is_valid:
                        buffered = buffered.buffer(0)
                    
                    buffered_geoms.append(buffered)
                except Exception as buffer_e:
                    emit_error(session_id, f"Error buffering geometry ID {geom}: {str(buffer_e)}", socketio)
                    continue

            # Merge all buffered geometries
            try:
                merged_buffer = unary_union(buffered_geoms)
                emit_progress(session_id, f"Merged buffers for {distance} meters.", socketio)
            except Exception as merge_e:
                emit_error(session_id, f"Error merging buffers for {distance} meters: {str(merge_e)}", socketio)
                continue

            # Check if merged_buffer is a supported geometry type
            if isinstance(merged_buffer, (Polygon, MultiPolygon, LineString, MultiLineString)):
                buffered_gdf = gpd.GeoDataFrame(geometry=[merged_buffer], crs=gdf.crs)
                buffered_gdf['buffer_distance'] = distance
                buffered_gdfs.append(buffered_gdf)
            else:
                emit_error(session_id, f"Merged buffer geometry type {type(merged_buffer)} is not supported.", socketio)
                continue
        return buffered_gdfs
    except Exception as e:
        emit_error(session_id, f"Error creating buffers: {str(e)}", socketio)
        return []

def fetch_and_buffer(engine, table_name, buffer_distances, session_id, socketio):
    """
    Fetch data from a specified table and create buffers.
    """
    query = f'SELECT id, geom FROM "{table_name}"'
    try:
        gdf = fetch_data_from_postgis(query, engine, session_id, socketio)
        if gdf is None:
            return None, None
        gdf_projected = reproject_to_nyeri(gdf, session_id, socketio)
        emit_progress(session_id, f"Geometry types for {table_name}: {gdf_projected.geometry.type.unique()}", socketio)
        
        buffers = create_arcgis_like_buffers(gdf_projected, buffer_distances, session_id, socketio)
        if not buffers:
            emit_error(session_id, f"No buffers created for {table_name}.", socketio)
            return gdf_projected, []
        return gdf_projected, buffers
    except Exception as e:
        emit_error(session_id, f"Error processing {table_name}: {str(e)}", socketio)
        return None, None

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
            dem_df = pd.read_sql(query, connection)
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

def create_dem_raster_rasterize(dem_gdf, resolution, session_id, socketio):
    """
    Rasterize DEM data.
    """
    try:
        bounds = dem_gdf.total_bounds
        width = int((bounds[2] - bounds[0]) / resolution)
        height = int((bounds[3] - bounds[1]) / resolution)
        transform = from_bounds(*bounds, width, height)
        
        emit_progress(session_id, f"DEM Bounds: {bounds}", socketio)
        emit_progress(session_id, f"Raster Dimensions: width={width}, height={height}", socketio)
        
        dem_raster = rasterize(
            [(geom, value) for geom, value in zip(dem_gdf.geometry, dem_gdf['elevation'])],
            out_shape=(height, width),
            transform=transform,
            fill=np.nan,
            dtype='float32'
        )
        emit_progress(session_id, "DEM rasterization successful.", socketio)
        return dem_raster, transform
    except Exception as e:
        emit_error(session_id, f"Error rasterizing DEM data: {str(e)}", socketio)
        return None, None

def calculate_slope(dem_raster, transform, session_id, socketio):
    """
    Calculate slope from DEM raster.
    """
    try:
        x, y = np.gradient(dem_raster, transform[0], transform[4])
        slope = np.sqrt(x**2 + y**2)
        emit_progress(session_id, "Slope calculation successful.", socketio)
        return slope
    except Exception as e:
        emit_error(session_id, f"Error calculating slope: {str(e)}", socketio)
        return None

def visualize_slope(slope, transform, session_id, socketio):
    """
    Visualize the slope data.
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.imshow(slope, cmap='terrain', extent=plotting_extent(slope, transform))
        fig.colorbar(cax, ax=ax, orientation='vertical')
        ax.set_title('Slope Map')
        ax.set_axis_off()
        
        os.makedirs('output', exist_ok=True)
        slope_plot_path = os.path.join('output', f'slope_map_session_{session_id}.png')
        plt.savefig(slope_plot_path)
        plt.close(fig)
        
        emit_progress(session_id, f"Slope map saved as {slope_plot_path}.", socketio)
        return slope_plot_path
    except Exception as e:
        emit_error(session_id, f"Error visualizing slope: {str(e)}", socketio)
        return None

def process_dem_and_slope(engine, session_id, socketio):
    """
    Process DEM data to calculate and visualize slope.
    """
    try:
        emit_progress(session_id, "Starting DEM and slope analysis.", socketio)
        
        dem_gdf = fetch_dem_data(engine, session_id, socketio)
        if dem_gdf is None or dem_gdf.empty:
            emit_error(session_id, "Failed to fetch DEM data. Skipping DEM and slope analysis.", socketio)
            return None

        dem_raster, dem_transform = create_dem_raster_rasterize(dem_gdf, resolution=30, session_id=session_id, socketio=socketio)
        if dem_raster is None:
            emit_error(session_id, "DEM rasterization failed. Skipping slope calculation.", socketio)
            return None

        # Fill NaN values in DEM raster
        def fill_nan(array):
            if np.isnan(array).all():
                return np.nan
            return np.nanmean(array)
        
        dem_raster_filled = generic_filter(dem_raster, fill_nan, size=3, mode='nearest')
        emit_progress(session_id, "NaN values in DEM raster have been filled.", socketio)

        slope = calculate_slope(dem_raster_filled, dem_transform, session_id, socketio)
        if slope is None:
            emit_error(session_id, "Slope calculation failed.", socketio)
            return None

        slope_plot_path = visualize_slope(slope, dem_transform, session_id, socketio)

        slope_raster_path = os.path.join('output', f'slope_map_session_{session_id}.tif')
        with rasterio.open(
            slope_raster_path,
            'w',
            driver='GTiff',
            height=slope.shape[0],
            width=slope.shape[1],
            count=1,
            dtype='float32',
            crs=NYERI_CRS,
            transform=dem_transform
        ) as dst:
            dst.write(slope, 1)
        
        emit_progress(session_id, f"Slope raster saved as {slope_raster_path}", socketio)
        emit_progress(session_id, "DEM and slope analysis completed successfully.", socketio)
        return slope_plot_path
    except Exception as e:
        emit_error(session_id, f"Error in DEM and slope analysis: {str(e)}", socketio)
        return None

def plot_buffers(buffered_gdfs, original_gdf, session_id, socketio, title='Buffers', feature_color='blue'):
    """
    Plot buffer zones and save the plot.
    """
    try:
        emit_progress(session_id, f"Starting to plot buffers for {title}.", socketio)
        fig, ax = plt.subplots(figsize=(15, 15))
        
        for i, buffered_gdf in enumerate(buffered_gdfs):
            alpha = min(0.1 + (0.2 * i), 1.0)  # Ensure alpha is within 0-1 range
            buffered_gdf.plot(ax=ax, edgecolor='none', facecolor=feature_color, alpha=alpha)
        
        if original_gdf is not None and not original_gdf.empty:
            original_gdf.plot(ax=ax, color=feature_color, linewidth=1)
        
        ax.set_title(title, fontsize=16)
        ax.set_axis_off()
        
        os.makedirs('output', exist_ok=True)
        plot_path = os.path.join('output', f'{title.lower().replace(" ", "_")}_session_{session_id}.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)
        
        emit_progress(session_id, f"Buffer zones plot for {title} saved as {plot_path}.", socketio)
        return plot_path
    except Exception as e:
        emit_error(session_id, f"Error plotting buffers for {title}: {str(e)}", socketio)
        return None

def run_full_spatial_operations(engine, session_id, socketio):
    """
    Execute all spatial operations: fetching, buffering, DEM processing, and slope calculation.
    """
    try:
        emit_progress(session_id, "Starting full spatial operations.", socketio)

        # Define buffer distances for each feature type
        river_distances = [200, 500, 1000, 1500]
        road_distances = [200, 500, 1000]
        protected_area_distances = [200, 500, 1000, 1500]
        settlement_distances = [200, 500, 1000, 1500]

        feature_types = [
            ('River', river_distances),
            ('Road', road_distances),
            ('ProtectedArea', protected_area_distances),
            ('Settlement', settlement_distances)
        ]

        all_buffers = []
        all_original = []
        buffer_images = {}

        for feature_type, distances in feature_types:
            emit_progress(session_id, f"Processing {feature_type} data.", socketio)
            gdf, buffers = fetch_and_buffer(engine, feature_type, distances, session_id, socketio)
            if gdf is not None and not gdf.empty and buffers:
                all_buffers.extend(buffers)
                all_original.append(gdf)

                # Plot individual buffer
                image_path = plot_buffers(
                    buffers, 
                    gdf, 
                    session_id, 
                    socketio, 
                    title=f'{feature_type} Buffers', 
                    feature_color=BUFFER_COLORS.get(feature_type, 'blue')
                )
                if image_path:
                    buffer_images[feature_type] = image_path
            else:
                emit_error(session_id, f"Failed to process {feature_type} data.", socketio)

        if all_buffers and all_original:
            emit_progress(session_id, "Creating combined buffer zones plot.", socketio)
            try:
                # Concatenate all original GeoDataFrames
                concatenated_original = pd.concat(all_original, ignore_index=True)
                
                # Ensure 'geometry' column exists
                if 'geometry' not in concatenated_original.columns:
                    concatenated_original = gpd.GeoDataFrame(concatenated_original, geometry='geom', crs=NYERI_CRS)
                
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

        # Process DEM and calculate slope
        emit_progress(session_id, "Starting DEM and slope analysis.", socketio)
        slope_image_path = process_dem_and_slope(engine, session_id, socketio)
        if slope_image_path:
            buffer_images['Slope'] = slope_image_path
        else:
            emit_error(session_id, "Failed to generate slope map.", socketio)

        # Emit buffer image paths to frontend
        socketio.emit('buffer_images', {'session_id': session_id, 'images': buffer_images}, room=session_id)

        emit_progress(session_id, "All spatial operations completed successfully.", socketio)
        socketio.emit('operation_completed', {'session_id': session_id, 'message': 'Operations completed successfully.'}, room=session_id)
        eventlet.sleep(0)
    except Exception as e:
        emit_error(session_id, f"Unexpected error during spatial operations: {str(e)}", socketio)