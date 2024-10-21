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
from sqlalchemy import text, inspect, create_engine
import json
from tabulate import tabulate
import traceback
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from rasterio.mask import mask
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patheffects as pe
from waste_collection_optimization import optimize_waste_collection

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Nyeri, Kenya is approximately in UTM zone 37S, Arc 1960
NYERI_CRS = CRS.from_epsg(21037)

# Define consistent colors for buffers
BUFFER_COLORS = {
    'River': '#66c2a5',
    'Road': '#fc8d62',
    'ProtectedArea': '#8da0cb',
    'Settlement': '#e78ac3'
}

# Suitability mapping functions
def river_suitability_mapping(distance):
    if pd.isna(distance):
        return np.nan
    if distance <= 200:
        return 1  # Not suitable
    elif 200 < distance <= 500:
        return 2  # Less suitable
    elif 500 < distance <= 1000:
        return 3  # Moderately Suitable
    elif 1000 < distance <= 1500:
        return 4  # Suitable
    else:
        return 5  # Highly suitable

def residential_area_suitability_mapping(distance):
    return river_suitability_mapping(distance)

def soil_suitability_mapping(soil_type):
    if pd.isna(soil_type) or soil_type is None:
        return np.nan  # No data
    soil_type = str(soil_type).lower()
    if 'sand' in soil_type:
        return 1  # Not suitable
    elif 'loam' in soil_type:
        return 2  # Less suitable
    elif 'silt' in soil_type:
        return 4  # Suitable
    elif 'clay' in soil_type:
        return 5  # Highly suitable
    else:
        return 3  # Moderately suitable

def road_suitability_mapping(distance):
    if pd.isna(distance):
        return np.nan
    if distance <= 200:
        return 1  # Not suitable
    elif 200 < distance <= 500:
        return 3  # Moderately suitable
    elif 500 < distance <= 1000:
        return 5  # Highly suitable
    elif 1000 < distance <= 1500:
        return 4  # Suitable
    else:
        return 2  # Less suitable

def settlement_suitability_mapping(distance):
    return river_suitability_mapping(distance)

def protectedarea_suitability_mapping(distance):
    if pd.isna(distance):
        return np.nan
    if distance <= 200:
        return 1  # Not suitable
    elif 200 < distance <= 500:
        return 2  # Less suitable
    elif 500 < distance <= 1000:
        return 3  # Moderately Suitable
    elif 1000 < distance <= 1500:
        return 4  # Suitable
    else:
        return 5  # Highly suitable

def geology_suitability_mapping(geology_type):
    if pd.isna(geology_type) or geology_type is None:
        return np.nan  # No data
    geology_type = str(geology_type).lower()
    if geology_type == 'ti':
        return 3  # Suitable
    elif geology_type == 'qv':
        return 4  # Highly suitable
    elif geology_type == 'qc':
        return 2  # Moderately suitable
    else:
        return 1  # Not suitable

def slope_suitability_mapping(slope_degree):
    if pd.isna(slope_degree):
        return np.nan
    if slope_degree <= 5:
        return 5  # Highly suitable (0-5 degrees)
    elif 5 < slope_degree <= 10:
        return 4  # Highly suitable (5-10 degrees)
    elif 10 < slope_degree <= 15:
        return 3  # Suitable (10-15 degrees)
    elif 15 < slope_degree <= 20:
        return 2  # Moderately suitable (15-20 degrees)
    else:
        return 1  # Not suitable (>20 degrees)

def land_use_suitability_mapping(land_use):
    if pd.isna(land_use) or land_use is None:
        return np.nan  # No data
    land_use_type = land_use_types.get(land_use, '').lower()
    if 'forests' in land_use_type or 'buildup' in land_use_type:
        return 1  # Not suitable
    elif 'farmlands' in land_use_type:
        return 2  # Less suitable
    elif 'bareland' in land_use_type:
        return 5  # Highly suitable
    else:
        return 3  # Moderately suitable

def reclassify_suitability(suitability_scores):
    reclassified = np.full_like(suitability_scores, np.nan, dtype=np.float32)
    mask = ~np.isnan(suitability_scores)
    reclassified[mask & (suitability_scores <= 20)] = 1  # Not suitable
    reclassified[mask & (suitability_scores > 20) & (suitability_scores <= 40)] = 2  # Less suitable
    reclassified[mask & (suitability_scores > 40) & (suitability_scores <= 60)] = 3  # Moderately suitable
    reclassified[mask & (suitability_scores > 60) & (suitability_scores <= 80)] = 4  # Suitable
    reclassified[mask & (suitability_scores > 80)] = 5  # Highly suitable
    return reclassified

def mask_raster_with_boundary(raster_data, transform, nyeri_gdf):
    """Mask the raster data with the Nyeri boundary."""
    # Convert the boundary to GeoJSON format
    nyeri_geojson = [nyeri_gdf.geometry.unary_union.__geo_interface__]
    
    # Mask the raster data
    with MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            height=raster_data.shape[0],
            width=raster_data.shape[1],
            count=1,
            dtype=raster_data.dtype,
            crs=NYERI_CRS,
            transform=transform
        ) as dataset:
            dataset.write(raster_data, 1)
            out_image, out_transform = mask(dataset, nyeri_geojson, crop=True, nodata=np.nan)
    return out_image[0],out_transform
            
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patheffects as pe

def create_suitability_map(data, title, output_path, transform, crs, nyeri_gdf):
    # Mask the data with the Nyeri boundary
    masked_data, masked_transform = mask_raster_with_boundary(data, transform, nyeri_gdf)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Define color scheme
    cmap = colors.ListedColormap(['purple', 'red', 'yellow', 'lightgreen', 'darkgreen'])
    norm = colors.BoundaryNorm([1, 2, 3, 4, 5, 6], cmap.N)
    
    # Add transparency for no data
    cmap.set_bad(color='white', alpha=0)
    
    extent = plotting_extent(masked_data, masked_transform)
    im = ax.imshow(masked_data, cmap=cmap, norm=norm, extent=extent)
    
    # Plot Nyeri boundary
    nyeri_gdf.boundary.plot(ax=ax, edgecolor='black', linewidth=1, label='Nyeri Boundary')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='purple', label='Not suitable'),
        mpatches.Patch(color='red', label='Less suitable'),
        mpatches.Patch(color='yellow', label='Moderately suitable'),
        mpatches.Patch(color='lightgreen', label='Suitable'),
        mpatches.Patch(color='darkgreen', label='Highly suitable')
    ]
    legend = ax.legend(handles=legend_elements, loc='lower left', fontsize=8, bbox_to_anchor=(0, -0.3))
    
    # Add map information
    map_info = ax.text(0.5, -0.3, 'Coordinate system: Arc 1960 UTM Zone 37S\nProjection: Transverse Mercator\nDatum: Arc 1960',
                       transform=ax.transAxes, fontsize=8, ha='right', va='top')
    
    ax.set_title(f'{title}', fontsize=16)
    ax.set_xlabel('Easting (meters)')
    ax.set_ylabel('Northing (meters)')
    
    # Add scale bar
    scalebar = ScaleBar(1, location='lower center', scale_loc='bottom', length_fraction=0.5, units='km', dimension='si-length', label='Scale')
    ax.add_artist(scalebar)
    
    # Add north arrow
    ax.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction', 
                horizontalalignment='center', verticalalignment='center',
                fontsize=16, fontweight='bold', path_effects=[pe.withStroke(linewidth=3, foreground="w")])
    ax.arrow(0.95, 0.93, 0, 0.02, head_width=0.01, head_length=0.01, 
             fc='k', ec='k', transform=ax.transAxes)
    
    # Add neatline
    fig.patch.set_linewidth(2)
    fig.patch.set_edgecolor('black')
    
    # Set gridlines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}'))
    
    # Adjust layout with padding
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.4)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
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

def fetch_data_from_postgis(query, engine, session_id, socketio):
    try:
        emit_progress(session_id, f"Executing query: {query}", socketio)
        gdf = gpd.read_postgis(query, engine, geom_col='geom')
        emit_progress(session_id, f"Fetched {len(gdf)} geometries.", socketio)
        return gdf
    except Exception as e:
        emit_error(session_id, f"Error fetching data: {str(e)}", socketio)
        return None

def fetch_nyeri_boundary(engine, session_id, socketio):
    try:
        emit_progress(session_id, "Fetching Nyeri boundary from the database.", socketio)
        query = """
        SELECT geom
        FROM "AreaOfInterest"
        """
        gdf = fetch_data_from_postgis(query, engine, session_id, socketio)
        if gdf is None or gdf.empty:
            raise ValueError("Nyeri boundary not found in AreaOfInterest table.")
        
        nyeri_gdf = reproject_to_nyeri(gdf, session_id, socketio)
        emit_progress(session_id, "Nyeri boundary fetched and reprojected.", socketio)
        return nyeri_gdf
    except Exception as e:
        emit_error(session_id, f"Error fetching Nyeri boundary: {str(e)}", socketio)
        return None

def reproject_to_nyeri(gdf, session_id, socketio):
    try:
        emit_progress(session_id, f"Original CRS: {gdf.crs}", socketio)
        emit_progress(session_id, f"Reprojecting data to Nyeri CRS (EPSG:21037)", socketio)
        gdf_projected = gdf.to_crs(NYERI_CRS)
        emit_progress(session_id, f"Data reprojected to Nyeri CRS (EPSG:21037)", socketio)
        return gdf_projected
    except Exception as e:
        emit_error(session_id, f"Error reprojecting data: {str(e)}", socketio)
        return None

def mask_to_nyeri(gdf, nyeri_gdf, session_id, socketio):
    try:
        emit_progress(session_id, "Masking GeoDataFrame to Nyeri boundary.", socketio)
        masked_gdf = gpd.overlay(gdf, nyeri_gdf, how='intersection')
        emit_progress(session_id, "Masking completed.", socketio)
        return masked_gdf
    except Exception as e:
        emit_error(session_id, f"Error masking GeoDataFrame to Nyeri: {str(e)}", socketio)
        return gpd.GeoDataFrame(columns=gdf.columns)

def create_arcgis_like_buffers(gdf, distances, session_id, socketio, nyeri_gdf):
    buffers = []
    for distance in distances:
        try:
            emit_progress(session_id, f"Creating buffer of {distance} meters.", socketio)
            buffer = gdf.geometry.buffer(distance)
            merged_buffer = gpd.GeoDataFrame(geometry=[buffer.unary_union], crs=gdf.crs)
            masked_buffer = mask_to_nyeri(merged_buffer, nyeri_gdf, session_id, socketio)
            buffers.append(masked_buffer)
            emit_progress(session_id, f"Merged and masked buffers for {distance} meters.", socketio)
        except Exception as e:
            emit_error(session_id, f"Error creating buffer at {distance} meters: {str(e)}", socketio)
    return buffers

def plot_buffers(buffers, original_gdf,nyeri_gdf, session_id, socketio, title='Buffer Zones'):
    try:
        emit_progress(session_id, f"Starting to plot buffers for {title}.", socketio)
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot Nyeri boundary
        nyeri_gdf.boundary.plot(ax=ax, edgecolor='black', linewidth=1, label='Nyeri Boundary')
        
        # Plot original geometries
        original_gdf.plot(ax=ax, color='red', alpha=0.5, label='Original Geometries')
        
        # Plot buffer zones
        for i, buffer_set in enumerate(buffers):
            buffer_set.plot(ax=ax, alpha=0.3, color=BUFFER_COLORS.get(title.split()[0], f'C{i}'), label=f'Buffer {i+1}')
        
        ax.set_title(title)
        ax.axis('off')
        ax.legend()

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
    query = """
    SELECT id, name, elevation, "geometryType", 
           ST_AsText(geom) as geom_wkt
    FROM "DigitalElevationModel"
    """
    try:
        emit_progress(session_id, "Fetching DEM data from the database.", socketio)
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

def fetch_and_classify_vector(table_name, classification_func, engine, session_id, socketio, nyeri_gdf):
    try:
        emit_progress(session_id, f"Fetching and classifying {table_name} data.", socketio)
        
        # Construct the query based on available columns
        if table_name == 'Geology':
            query = f'SELECT id, geom, properties FROM "{table_name}"'
        elif table_name == 'Soil':
            query = f'SELECT id, geom, "soilType" FROM "{table_name}"'
        else:
            raise ValueError(f"Unsupported table: {table_name}")
        
        gdf = fetch_data_from_postgis(query, engine, session_id, socketio)
        
        if gdf is not None:
            gdf_projected = reproject_to_nyeri(gdf, session_id, socketio)
            gdf_masked = mask_to_nyeri(gdf_projected, nyeri_gdf, session_id, socketio)
            
            if 'geom' in gdf_masked.columns:
                gdf_masked = gdf_masked.rename(columns={'geom': 'geometry'}).set_geometry('geometry')
            
            geometry_types = gdf_masked.geometry.geom_type.unique()
            emit_progress(session_id, f"Geometry types for {table_name}: {geometry_types}", socketio)
            
            if table_name == 'Geology':
                # Extract 'GLG' from properties JSON
                gdf_masked['classification_field'] = gdf_masked['properties'].apply(lambda x: json.loads(x)['GLG'] if isinstance(x, str) else x.get('GLG') if isinstance(x, dict) else None)
            elif table_name == 'Soil':
                gdf_masked['classification_field'] = gdf_masked['soilType']
            
            gdf_masked['suitability_score'] = gdf_masked['classification_field'].apply(classification_func)
            emit_progress(session_id, f"Fetched and classified {table_name} data.", socketio)
            return gdf_masked
        else:
            emit_error(session_id, f"No data fetched for {table_name}.", socketio)
            return None
    except Exception as e:
        emit_error(session_id, f"Error fetching and classifying {table_name} data: {str(e)}", socketio)
        return None

def fetch_and_plot_landuse_raster(engine, session_id, socketio):
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
    try:
        def landuse_suitability_mapping_internal(land_use):
            land_use_type = land_use_types.get(land_use, '').lower()
            if 'forests' in land_use_type:
                return 1  # Not suitable
            elif 'bareland' in land_use_type:
                return 5  # Highly suitable
            elif 'buildup' in land_use_type:
                return 2  # Less suitable
            elif 'farmland' in land_use_type:
                return 4  # Suitable
            else:
                return np.nan  # No Data or Unknown land use type
        
        landuse_suitability = np.vectorize(landuse_suitability_mapping_internal)(landuse_raster)
        emit_progress(session_id, "Land use suitability mapping applied.", socketio)
        return landuse_suitability
    except Exception as e:
        emit_error(session_id, f"Error processing land use suitability: {str(e)}", socketio)
        return None

def train_suitability_model(X, y, session_id, socketio):
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
    criteria = [
        'River',
        'Residential Area',
        'Soil',
        'Road',
        'Settlement',
        'ProtectedArea',
        'Geology',
        'LandUse'
    ]
    n = len(criteria)
    
    # Example pairwise comparison matrix (AHP)
    # This should be filled based on actual pairwise comparisons
    # Here, it's assumed to be a consistent matrix for demonstration
    matrix = np.array([
        [1,     3,      5,      5,      3,      3,      5,      5],  # River
        [1/3,   1,      3,      3,      3,      3,      3,      3],  # Residential Area
        [1/5,   1/3,    1,      1,      1,      1,      1,      1],  # Soil
        [1/5,   1/3,    1,      1,      1,      1,      1,      1],  # Road
        [1/3,   1/3,    1,      1,      1,      1,      1,      1],  # Settlement
        [1/3,   1/3,    1,      1,      1,      1,      1,      1],  # ProtectedArea
        [1/5,   1/3,    1,      1,      1,      1,      1,      1],  # Geology
        [1/5,   1/3,    1,      1,      1,      1,      1,      1],  # LandUse
    ])

    try:
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        max_index = np.argmax(eigenvalues.real)
        weights = eigenvectors[:, max_index].real
        weights = weights / np.sum(weights)

        weights_percent = weights * 100

        weights_dict = dict(zip(criteria, weights_percent))

        return weights_dict, matrix
    except Exception as e:
        print(f"Error calculating weights: {e}")
        return {}, matrix

def calculate_consistency_ratio(matrix, weights):
    try:
        n = matrix.shape[0]
        weighted_sum = np.dot(matrix, weights)
        lambda_max = np.sum(weighted_sum / weights) / n

        consistency_index = (lambda_max - n) / (n - 1)

        random_index = {3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24,
        7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
        ri = random_index.get(n, 1.49)  # Default to 1.49 if n > 10

        consistency_ratio = consistency_index / ri

        return consistency_ratio
    except Exception as e:
        print(f"Error calculating consistency ratio: {e}")
        return None

def rasterize_buffers(gdf, buffers, transform, shape, suitability_mapping, distances, session_id, socketio):
    try:
        emit_progress(session_id, "Rasterizing buffers.", socketio)
        
        # Initialize raster with NaNs
        buffer_raster = np.full(shape, np.nan, dtype='float32')
        
        # Rasterize each buffer with its suitability score
        for buffer, distance in zip(buffers, distances):
            value = suitability_mapping(distance)
            rasterized = rasterize(
                [(geom, value) for geom in buffer.geometry],
                out_shape=shape,
                transform=transform,
                fill=np.nan,
                all_touched=True,
                dtype='float32'
            )
            # Update buffer_raster where NaN
            buffer_raster = np.where(np.isnan(buffer_raster), rasterized, buffer_raster)
        # Fill remaining NaNs with the suitability score for areas beyond the maximum buffer distance
        buffer_raster = np.where(np.isnan(buffer_raster), suitability_mapping(max(distances) + 1), buffer_raster)
        
        emit_progress(session_id, "Buffers rasterized successfully.", socketio)
        return buffer_raster
    except Exception as e:
        emit_error(session_id, f"Error in rasterize_buffers: {str(e)}", socketio)
        return None

def run_full_spatial_operations(engine, session_id, socketio):
    try:
        emit_progress(session_id, "Initiating full spatial analysis for landfill site suitability.", socketio)
        
        # Fetch Nyeri boundary
        nyeri_gdf = fetch_nyeri_boundary(engine, session_id, socketio)
        if nyeri_gdf is None:
            raise ValueError("Nyeri boundary could not be fetched.")
        
        weights_dict, matrix = calculate_weights()
        emit_progress(session_id, f"AHP Weights calculated: {weights_dict}", socketio)
        
        consistency_ratio = calculate_consistency_ratio(matrix, np.array(list(weights_dict.values())))
        emit_progress(session_id, f"Consistency Ratio: {consistency_ratio:.4f}", socketio)
        if consistency_ratio < 0.1:
            emit_progress(session_id, "AHP matrix is consistent (CR < 0.1).", socketio)
        else:
            emit_progress(session_id, "Warning: The pairwise comparison matrix is not consistent (CR >= 0.1).", socketio)
            emit_progress(session_id, "Please revise the comparison matrix to improve consistency.", socketio)
        
        feature_types = [
            ('River', [200, 500, 1000, 1500]),
            ('Road', [200, 500, 1000, 1500]),
            ('ProtectedArea', [200, 500, 1000, 1500]),
            ('Settlement', [200, 500, 1000, 1500])
        ]
        
        all_buffers = []
        all_original = []
        buffer_images = {}
        
        for feature_type, distances in feature_types:
            emit_progress(session_id, f"Processing {feature_type} data.", socketio)
            
            query = f'SELECT id, geom FROM "{feature_type}"'
            gdf = fetch_data_from_postgis(query, engine, session_id, socketio)
            
            if gdf is not None:
                gdf_projected = reproject_to_nyeri(gdf, session_id, socketio)
                gdf_masked = mask_to_nyeri(gdf_projected, nyeri_gdf, session_id, socketio)
                
                if 'geometry' in gdf_masked.columns:
                    gdf_masked = gdf_masked.rename(columns={'geom': 'geometry'}).set_geometry('geometry')
                
                geometry_types = gdf_masked.geometry.geom_type.unique()
                emit_progress(session_id, f"Geometry types for {feature_type}: {geometry_types}", socketio)
                
                buffers = create_arcgis_like_buffers(gdf_masked, distances, session_id, socketio, nyeri_gdf)
                
                if buffers:
                    all_buffers.append(buffers)
                    all_original.append(gdf_masked)
                    
                    buffer_plot_path = plot_buffers(buffers, gdf_masked, nyeri_gdf, session_id, socketio, title=f'{feature_type} Buffers')
                    if buffer_plot_path:
                        buffer_images[feature_type] = buffer_plot_path
        
        if all_buffers and all_original:
            emit_progress(session_id, "Creating combined buffer zones plot.", socketio)
            try:
                concatenated_original = pd.concat(all_original, ignore_index=True)
                
                
                all_buffers_combined = gpd.GeoDataFrame(pd.concat(
                    [gdf for buffer_list in all_buffers for gdf in buffer_list], ignore_index=True
                ))        
                               # Ensure the combined GeoDataFrame has a geometry column
                if 'geometry' not in all_buffers_combined.columns:
                 all_buffers_combined = all_buffers_combined.set_geometry('geometry')
     
                emit_progress(session_id, f"Data types: {concatenated_original.dtypes}", socketio)
                
                if 'geometry' in concatenated_original.columns:
                    concatenated_original = gpd.GeoDataFrame(concatenated_original, geometry='geometry', crs=NYERI_CRS)
                else:
                    raise ValueError("No 'geometry' column found in concatenated data.")
                
                combined_image_path = plot_buffers(
                    all_buffers_combined,
                    nyeri_gdf,
                    concatenated_original, 
                    session_id, 
                    socketio, 
                    title='Combined Buffer Zones'
                )
                if combined_image_path:
                    buffer_images['Combined'] = combined_image_path
                emit_progress(session_id, f"Combined buffer zones plot created: {combined_image_path}", socketio)
            except Exception as e:
                emit_error(session_id, f"Error creating combined buffer zones: {str(e)}", socketio)
        else:
            emit_error(session_id, "No valid data to create combined buffer zones.", socketio)
        
        emit_progress(session_id, "Starting DEM and slope analysis.", socketio)
        dem_gdf = fetch_dem_data(engine, session_id, socketio)
        if dem_gdf is not None:
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
                all_touched=True,
                dtype='float32'
            )
            
            emit_progress(session_id, f"DEM Raster Shape: {dem_raster.shape}", socketio)
            emit_progress(session_id, f"DEM Raster Stats: min={np.nanmin(dem_raster)}, max={np.nanmax(dem_raster)}, mean={np.nanmean(dem_raster)}", socketio)
            
            # Fill NaN values with nearest valid 
            from scipy.ndimage import generic_filter
            
            def fill_nan(array):
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
            buffer_images['Slope'] = slope_plot_path
        else:
                emit_error(session_id, "Slope calculation failed.", socketio)
        
        geology_gdf = fetch_and_classify_vector('Geology', geology_suitability_mapping, engine, session_id, socketio, nyeri_gdf)
        soil_gdf = fetch_and_classify_vector('Soil', soil_suitability_mapping, engine, session_id, socketio, nyeri_gdf)
        
        emit_progress(session_id, "Starting land use processing", socketio)
        landuse_raster, landuse_transform, landuse_crs, landuse_plot_path, land_use_types = fetch_and_plot_landuse_raster(engine, session_id, socketio)
        if landuse_raster is not None:
            buffer_images['LandUse'] = landuse_plot_path
            landuse_suitability = process_landuse_suitability(landuse_raster, land_use_types, session_id, socketio)
            emit_progress(session_id, "Land use suitability processing completed", socketio)
        else:
            emit_error(session_id, "Failed to fetch land use raster.", socketio)
            landuse_suitability = None
        
        emit_progress(session_id, "Starting to create common grid for all layers", socketio)
        if all_original:
            emit_progress(session_id, "all_original is not empty, proceeding with grid creation", socketio)
            try:
                bounds = concatenated_original.total_bounds
                emit_progress(session_id, f"Common grid bounds calculated: {bounds}", socketio)
            except Exception as e:
                emit_error(session_id, f"Error calculating bounds: {str(e)}", socketio)
                raise

            res = 30  # 30m resolution, adjust as needed
            rows = int((bounds[3] - bounds[1]) / res)
            cols = int((bounds[2] - bounds[0]) / res)
            transform = from_bounds(*bounds, cols, rows)
            emit_progress(session_id, f"Created transform with rows={rows}, cols={cols}", socketio)

            layers = {}
            emit_progress(session_id, "Rasterizing vector layers to common grid...", socketio)
            for i, (feature_type, distances) in enumerate(feature_types):
                try:
                    suitability_mapping = globals()[f"{feature_type.lower()}_suitability_mapping"]
                    layers[f'Buffer_{feature_type}'] = rasterize_buffers(
                        all_original[i],
                        all_buffers[i],
                        transform,
                        (rows, cols),
                        suitability_mapping,
                        distances,
                        session_id,
                        socketio
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
                        fill=np.nan,
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
                        fill=np.nan,
                        dtype='float32'
                    )
                    emit_progress(session_id, "Rasterized Soil layer", socketio)
                except Exception as e:
                    emit_error(session_id, f"Error rasterizing Soil layer: {str(e)}", socketio)

            if landuse_suitability is not None:
                try:
                    resampled_landuse = np.full((rows, cols), np.nan, dtype='float32')
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

            if slope is not None:
                try:
                    resampled_slope = np.full((rows, cols), np.nan, dtype='float32')
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
                    emit_progress(session_id, "Slope layer resampled and added to common grid", socketio)
                except Exception as e:
                    emit_error(session_id, f"Error adding Slope layer: {str(e)}", socketio)

            emit_progress(session_id, "Starting weighted suitability calculation", socketio)
            total_suitability = np.full((rows, cols), np.nan)
            for layer_name, layer in layers.items():
                if layer_name in weights_dict:
                    weight = weights_dict[layer_name]
                elif layer_name.startswith('Buffer_'):
                    weight = weights_dict.get('Buffers', 1) / len([l for l in layers.keys() if l.startswith('Buffer_')])
                else:
                    weight = weights_dict.get(layer_name, 0)
                
                # Use np.nansum to ignore NaN values during addition
                total_suitability = np.nansum([total_suitability, layer * weight], axis=0)
                emit_progress(session_id, f"Added {layer_name} to total suitability", socketio)

            # Normalize only valid data
            valid_mask = ~np.isnan(total_suitability)
            total_suitability[valid_mask] = ((total_suitability[valid_mask] - np.nanmin(total_suitability)) / 
                                             (np.nanmax(total_suitability) - np.nanmin(total_suitability))) * 100
            emit_progress(session_id, "Total suitability scores normalized to 0-100 range", socketio)

            reclassified_suitability = reclassify_suitability(total_suitability)
        
            for layer_name, layer in layers.items():
            # Remove "Buffer_" from the layer name
                clean_layer_name = layer_name.replace('Buffer_', '')
    
                output_path = os.path.join('output', f'{layer_name.lower()}_suitability_map_session_{session_id}.png')
                create_suitability_map(layer, f'{clean_layer_name} Suitability Map', output_path, transform, NYERI_CRS, nyeri_gdf)
    
                buffer_images[f'{layer_name}Suitability'] = output_path
                emit_progress(session_id, f"Created suitability map for {clean_layer_name}", socketio)

            output_path = os.path.join('output', f'total_suitability_map_session_{session_id}.png')
            create_suitability_map(reclassified_suitability, 'Total Suitability Map', output_path, transform, NYERI_CRS, nyeri_gdf)
            buffer_images['TotalSuitability'] = output_path
            emit_progress(session_id, "Created total suitability map", socketio)

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

            X = total_suitability.reshape(-1, 1)
            y = (total_suitability > np.nanpercentile(total_suitability, 75)).astype(int).ravel()
            
            model, scaler = train_suitability_model(X[~np.isnan(X).ravel()], y[~np.isnan(X).ravel()], session_id, socketio)
            
            if model is not None and scaler is not None:
                emit_progress(session_id, "Applying machine learning model...", socketio)
                ml_suitability_map_path, ml_suitability_raster_path = apply_ml_model(
                    total_suitability, model, scaler, rows, cols, transform, session_id, socketio
                )
                
                if ml_suitability_map_path:
                    buffer_images['MLSuitability'] = ml_suitability_map_path
                    
            emit_progress(session_id, "Starting waste collection optimization", socketio)

            result = optimize_waste_collection(engine, session_id, socketio, nyeri_gdf)

            if result is None:
                emit_progress(session_id, "Waste collection optimization failed or returned no results.", socketio)
            else:
                optimal_locations, plot_paths = result
                
                if optimal_locations is not None and not optimal_locations.empty:
                    emit_progress(session_id, f"Waste collection optimization completed. Found {len(optimal_locations)} optimal locations.", socketio)
                    # Add the optimal locations plot to the buffer_images dictionary
                    buffer_images['OptimalWasteCollectionPoints'] = plot_paths.get('optimal_locations')
                    # Add other waste collection plots to the buffer_images dictionary
                    buffer_images['WasteCollectionRoadSuitability'] = plot_paths.get('road_suitability')
                    buffer_images['WasteCollectionSettlementSuitability'] = plot_paths.get('settlement_suitability')
                    buffer_images['WasteCollectionCombinedSuitability'] = plot_paths.get('combined_suitability')
                else:
                    emit_progress(session_id, "Waste collection optimization completed but found no optimal locations.", socketio)
            
            socketio.emit('buffer_images', {'session_id': session_id, 'images': buffer_images}, room=session_id)

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
    engine = create_engine('postgresql://username:password@host:port/database')
    
    class MockSocketIO:
        def emit(self, event, data, room=None):
            print(f"Emitted {event}: {data}")
    
    mock_socketio = MockSocketIO()
    run_full_spatial_operations(engine, "test_session", mock_socketio)