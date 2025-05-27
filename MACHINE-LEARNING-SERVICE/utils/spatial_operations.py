import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from rasterio.plot import plotting_extent
from rasterio.mask import mask
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patheffects as pe
from shapely.geometry import box, MultiPolygon, LineString, Polygon,Point, MultiLineString
from shapely.wkt import loads
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
import numpy.ma as ma
import rasterio.warp
from waste_collection_optimization import optimize_waste_collection
from scipy.ndimage import sobel
from matplotlib.lines import Line2D
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import joblib
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from utils.predict_suitability import predict_map_suitability

from utils.create_training_dataset import create_training_dataset
import matplotlib.lines as mlines
from utils.network_analysis import perform_network_analysis


# # Import the grid analysis functions
# from utils.grid_analysis import (
#     grid_based_suitability_analysis,
#     create_prediction_map,
#     create_completeness_map,
#     create_distribution_plots,
#     perform_sensitivity_analysis,
#     plot_sensitivity_results,
#     create_interactive_map
# )

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
    if distance <= 300:
        return 1  # Not suitable
    elif 300 < distance <= 1000:
        return 2  # Less suitable
    elif 1000 < distance <= 1500:
        return 3  # Moderately suitable
    elif 1500 < distance <= 2000:
        return 4  # Suitable
    else:
        return 5  # Highly suitable

def road_suitability_mapping(distance):
    if pd.isna(distance):
        return np.nan
    if distance > 1200:
        return 1  # Not suitable
    elif 1000 < distance <= 1200:
        return 2  # Less suitable
    elif 800 < distance <= 1000:
        return 3  # Moderately suitable
    elif 400 < distance <= 800:
        return 4  # Suitable
    else:
        return 5  # Highly suitable

def settlement_suitability_mapping(distance):
    if pd.isna(distance):
        return np.nan
    if distance <= 400:
        return 1  # Not suitable
    elif 400 < distance <= 900:
        return 2  # Less suitable
    elif 1500 < distance <= 2100:
        return 3  # Moderately suitable
    elif distance > 2100:
        return 4  # Suitable
    else:
        return 5  # Highly 
    
def protectedarea_suitability_mapping(distance):
    if pd.isna(distance):
        return np.nan
    if distance <= 300:
        return 1  # Not suitable
    elif 300 < distance <= 1000:
        return 2  # Less suitable
    elif 1000 < distance <= 1500:  # Fixed the gap
        return 3  # Moderately suitable
    elif 1500 < distance <= 2000:
        return 4  # Suitable
    else:
        return 5  # Highly suitable

def slope_suitability_mapping(slope_degree):
    if pd.isna(slope_degree):
        return np.nan
    if slope_degree > 18.6:
        return 1  # Not suitable
    elif 5.3 < slope_degree <= 18.6:
        return 2  # Less suitable
    elif 3.1 < slope_degree <= 5.2:
        return 3  # Moderately suitable
    elif 1.7 < slope_degree <= 3:
        return 4  # Suitable
    else:
        return 5  # Highly suitable

def land_use_suitability_mapping(land_use):
    if pd.isna(land_use) or land_use is None:
        return np.nan  # No data
    if 'forests' in land_use or 'settlements' in land_use:
        return 1  # Not suitable
    elif 'farmlands' in land_use:
        return 4  # Suitable
    elif 'bareland' in land_use:
        return 5  # Highly suitable
    else:
        return 2  # Less 
def soil_suitability_mapping(soil_type):
    """Map soil types to suitability scores based on the actual database values"""
    try:
        if pd.isna(soil_type) or soil_type is None:
            print(f"Soil type is None or NaN")
            return np.nan
        
        soil_type = str(soil_type).strip()
        print(f"Processing soil type: {soil_type}")
        
        # Define exact soil type mappings based on your database values
        soil_mappings = {
            'Bk31-2a': 4,  # Calcic Cambisols - Less suitable
            'Ne12-2c': 3,  # Eutric Nitosols - Moderately suitable
            'Nh2-2c': 4    # Humic Nitosols - Less suitable
        }
        
        # Check for exact matches
        if soil_type in soil_mappings:
            suitability = soil_mappings[soil_type]
            print(f"Matched soil type '{soil_type}' with suitability {suitability}")
            return suitability
        
        # Handle variants
        if soil_type.startswith('Bk'):
            print(f"Matched Calcic Cambisols variant '{soil_type}' - Less suitable")
            return 2
        elif soil_type.startswith('Ne'):
            print(f"Matched Eutric Nitosols variant '{soil_type}' - Moderately suitable")
            return 3
        elif soil_type.startswith('Nh'):
            print(f"Matched Humic Nitosols variant '{soil_type}' - Less suitable")
            return 2
        
        # Default case
        print(f"No match found for '{soil_type}', defaulting to moderately suitable")
        return 3
        
    except Exception as e:
        print(f"Error in soil_suitability_mapping: {str(e)}")
        return np.nane

        
# def geology_suitability_mapping(geology_type):
#     if pd.isna(geology_type) or geology_type is None:
#         return np.nan  # No data
#     geology_type = str(geology_type).lower()
#     if geology_type == 'ti':
#         return 3  # Suitable
#     elif geology_type == 'qv':
#         return 4  # Highly suitable
#     elif geology_type == 'qc':
#         return 2  # Moderately suitable
#     else:
#         return 1  # Not suitable

def reclassify_suitability(suitability_scores):
    reclassified = np.full_like(suitability_scores, np.nan, dtype=np.float32)
    mask = ~np.isnan(suitability_scores)
    reclassified[mask & (suitability_scores <= 20)] = 1
    reclassified[mask & (suitability_scores > 20) & (suitability_scores <= 40)] = 2
    reclassified[mask & (suitability_scores > 40) & (suitability_scores <= 60)] = 3
    reclassified[mask & (suitability_scores > 60) & (suitability_scores <= 80)] = 4
    reclassified[mask & (suitability_scores > 80)] = 5
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
    return out_image[0], out_transform

def create_scale_bar(ax, length=10, units='km', subdivisions=5):
    # Create the main rectangle
    rect = mpatches.Rectangle((0, 0), length, 0.5, facecolor='white', edgecolor='black')
    ax.add_patch(rect)
    
    # Add subdivisions
    for i in range(1, subdivisions):
        x = i * (length / subdivisions)
        ax.plot([x, x], [0, 0.5], color='black', linewidth=1)
    
    # Add labels
    for i in range(subdivisions + 1):
        x = i * (length / subdivisions)
        ax.text(x, -0.25, str(int(x)), ha='center', va='top', fontsize=8)
    
    # Add unit label
    ax.text(length / 2, 0.75, units, ha='center', va='bottom', fontsize=8)
    
    # Set limits and remove axes
    ax.set_xlim(0, length)
    ax.set_ylim(-0.5, 1)
    ax.axis('off')


def create_suitability_map(data, title, output_path, transform, crs, nyeri_gdf, plot_boundary=True):
    try:
        # Mask the data with the Nyeri boundary
        masked_data, masked_transform = mask_raster_with_boundary(data, transform, nyeri_gdf)
        fig, ax = plt.subplots(figsize=(14, 14))
        # Define color scheme
        cmap = colors.ListedColormap(['purple', 'red', 'yellow', 'lightgreen', 'darkgreen'])
        norm = colors.BoundaryNorm([1, 2, 3, 4, 5, 6], cmap.N)
        # Add transparency for no data
        cmap.set_bad(color='white', alpha=0)
        extent = plotting_extent(masked_data, masked_transform)
        im = ax.imshow(masked_data, cmap=cmap, norm=norm, extent=extent)
        # Plot roads if title contains "road"
        if 'road' in title.lower():
            try:

                connection_params = {
                 'dbname': 'wms',
                  'user': 'postgres',
                  'password': '',
                  'host': 'localhost',
                  'port': '5432'
                }
                
                #  connection string
                conn_string = f"postgresql://{connection_params['user']}:{connection_params['password']}@{connection_params['host']}:{connection_params['port']}/{connection_params['dbname']}"
                
                # Query to fetch roads
                query = 'SELECT geom FROM "Road"'
                with create_engine(conn_string).connect() as conn:
                    roads_gdf = gpd.read_postgis(query, conn, geom_col='geom')
                
                # Reproject roads to match the map CRS
                roads_gdf = roads_gdf.to_crs(crs)
                
                # Plot roads with a distinctive style
                roads_gdf.plot(ax=ax, color='black', linewidth=0.8, 
                             alpha=0.7, label='Roads',
                             zorder=5)  # zorder ensures roads are plotted on top
            except Exception as e:
                print(f"Error plotting roads: {e}")
                traceback.print_exc()
        
        # Plot Nyeri boundary only if plot_boundary is True
        if plot_boundary:
            nyeri_gdf.boundary.plot(ax=ax, edgecolor='black', linewidth=1, label='Nyeri Boundary')
        
        # Set title with adjusted y position
        ax.set_title(f'{title}', fontsize=18, fontweight='bold', y=1.05)
        
        ax.set_xlabel('Easting (meters)', fontsize=12)
        ax.set_ylabel('Northing (meters)', fontsize=12)
        
        # Add north arrow
        ax.annotate('N', xy=(0.98, 0.98), xycoords='axes fraction', 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=18, fontweight='bold', path_effects=[pe.withStroke(linewidth=3, foreground="w")])
        ax.arrow(0.98, 0.96, 0, 0.02, head_width=0.01, head_length=0.01, 
                 fc='k', ec='k', transform=ax.transAxes)
        
        # Set gridlines and ticks
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}'))
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(top=True, bottom=True, left=True, right=True, 
                      labeltop=True, labelbottom=True, labelleft=True, labelright=True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Create new axes for the bottom row
        bottom_ax = fig.add_axes([0.1, 0.02, 0.8, 0.15])
        bottom_ax.axis('off')
        
        # Create legend elements
        legend_elements = [
            mpatches.Patch(color='purple', label='Not suitable'),
            mpatches.Patch(color='red', label='Less suitable'),
            mpatches.Patch(color='yellow', label='Moderately suitable'),
            mpatches.Patch(color='lightgreen', label='Suitable'),
            mpatches.Patch(color='darkgreen', label='Highly suitable')
        ]
        
        # Add roads to legend if they were plotted
        if 'road' in title.lower():
            legend_elements.append(mlines.Line2D([], [], color='black', 
                                               linewidth=1, label='Roads'))
        
        # Add legend
        legend = bottom_ax.legend(handles=legend_elements, 
                                loc='center left', 
                                fontsize=10, 
                                bbox_to_anchor=(0, 0.5))
        
        # Add scale bar
        scale_ax = fig.add_axes([0.4, 0.05, 0.2, 0.03])
        create_scale_bar(scale_ax, length=10, units='km', subdivisions=5)
        
        # Add map information
        info_text = 'Coordinate system: Arc 1960 UTM Zone 37S\nProjection: Transverse Mercator\nDatum: Arc 1960'
        bottom_ax.text(1, 0.5, info_text, ha='right', va='center', fontsize=10, transform=bottom_ax.transAxes)
        
        # Adjust subplot to make room for title and bottom information
        plt.subplots_adjust(top=0.95, bottom=0.2)
        
        # Add neatline
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
        
        # Save the figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5, 
                   facecolor='white', edgecolor='black')
        plt.close(fig)
        
    except Exception as e:
        print(f"Error in create_suitability_map: {e}")
        traceback.print_exc()

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
        
       
        emit_progress(session_id, f"Type of fetched data: {type(gdf)}", socketio)
        
        if gdf is None or gdf.empty:
            raise ValueError("Nyeri boundary not found in AreaOfInterest table.")
        
        nyeri_gdf = reproject_to_nyeri(gdf, session_id, socketio)
        
        # Add more debug logging
        emit_progress(session_id, f"Type after reprojection: {type(nyeri_gdf)}", socketio)
        
        # Verify that nyeri_gdf is a GeoDataFrame
        if not isinstance(nyeri_gdf, gpd.GeoDataFrame):
            raise ValueError(f"Expected GeoDataFrame, got {type(nyeri_gdf)}")
            
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

def plot_buffers(buffers, original_gdf, nyeri_gdf, session_id, socketio, title='Buffer Zones'):
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
        # Handle masked arrays
        if isinstance(dem_raster, ma.MaskedArray):
            dem_data = dem_raster.filled(np.nan)
        else:
            dem_data = dem_raster

        # Calculate gradients
        dx = sobel(dem_data, axis=1) / (8 * transform[0])
        dy = sobel(dem_data, axis=0) / (8 * transform[4])

        # Calculate slope in radians
        slope_rad = np.arctan(np.sqrt(dx*dx + dy*dy))
        
        # Convert to degrees
        slope_deg = np.degrees(slope_rad)

        # If original data was masked, apply the same mask to the slope
        if isinstance(dem_raster, ma.MaskedArray):
            slope_deg = ma.masked_array(slope_deg, mask=dem_raster.mask)

        emit_progress(session_id, "Slope calculation successful.", socketio)
        return slope_deg
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

def calculate_weights():
    criteria = [
        'River',
        'Road',
        'Settlement',
        'Soil',
        'Protected Areas',
        'Land Use',
        'Slope'
    ]
    
    # Pre-calculated weights from our AHP analysis
    weights_dict = {
        'River': 36.54,
        'Road': 25.86,
        'Settlement': 17.97,
        'Soil': 9.24,
        'Protected Areas': 4.75,
        'Land Use': 3.30,
        'Slope': 2.34
    }
    
    # Updated pairwise comparison matrix
    matrix = np.array([
        [1,    2,    3,    5,    7,    8,    9],    # River
        [1/2,  1,    2,    4,    6,    7,    8],    # Road
        [1/3,  1/2,  1,    3,    5,    6,    7],    # Settlement
        [1/5,  1/4,  1/3,  1,    3,    4,    5],    # Soil
        [1/7,  1/6,  1/5,  1/3,  1,    2,    3],    # Protected Areas
        [1/8,  1/7,  1/6,  1/4,  1/2,  1,    2],    # Land Use
        [1/9,  1/8,  1/7,  1/5,  1/3,  1/2,  1]     # Slope
    ])

    return weights_dict, matrix

def calculate_consistency_ratio(matrix, weights):
    try:
        n = matrix.shape[0]
        # Convert weights dictionary values to array in the same order as matrix
        weights_array = np.array(list(weights.values())) / 100  # Convert percentages to decimals
        
        weighted_sum = np.dot(matrix, weights_array)
        lambda_max = np.sum(weighted_sum / weights_array) / n

        consistency_index = (lambda_max - n) / (n - 1)

        random_index = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 
                       6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
        ri = random_index.get(n, 1.49)

        consistency_ratio = consistency_index / ri

        return consistency_ratio, consistency_index, lambda_max
    except Exception as e:
        print(f"Error calculating consistency ratio: {e}")
        return None, None, None

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
    
def save_raster_as_tif(data, transform, crs, output_path, session_id, socketio):
    try:
        # Convert backslashes to forward slashes in the output path
        output_path = output_path.replace('\\', '/')
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(data, 1)
        emit_progress(session_id, f"Saved TIF file: {output_path}", socketio)
    except Exception as e:
        emit_error(session_id, f"Error saving TIF file {output_path}: {str(e)}", socketio)

def calculate_weighted_suitability(layers, weights_dict, session_id, socketio):
    """
    Calculate the weighted suitability by combining all layers according to their weights.
    
    Args:
    layers (dict): Dictionary containing paths to all raster layers
    weights_dict (dict): Dictionary containing weights for each layer
    session_id (str): Unique identifier for the current session
    socketio: SocketIO instance for real-time communication
    
    Returns:
    numpy.ndarray: The calculated total suitability raster
    """
    try:
        emit_progress(session_id, "Starting weighted suitability calculation", socketio)
        
        total_suitability = None
        total_weight = 0
        
        for layer_name, layer_info in layers.items():
            # Determine weight for the current layer
            if layer_name in weights_dict:
                weight = weights_dict[layer_name]
            elif layer_name.startswith('Buffer_'):
                weight = weights_dict.get('Buffers', 1) / len([l for l in layers.keys() if l.startswith('Buffer_')])
            else:
                weight = weights_dict.get(layer_name, 0)
            
            total_weight += weight
            
            # Read and process each layer
            with rasterio.open(layer_info['path']) as src:
                layer_data = src.read(1).astype(float)
                
                # Add debug information
                emit_progress(session_id, f"Layer {layer_name} stats: min={np.nanmin(layer_data)}, max={np.nanmax(layer_data)}, mean={np.nanmean(layer_data)}, NaN count={np.isnan(layer_data).sum()}", socketio)
                
                # Normalize layer data to 0-1 range
                min_val = np.nanmin(layer_data)
                max_val = np.nanmax(layer_data)
                if min_val != max_val:
                    layer_data = (layer_data - min_val) / (max_val - min_val)
                else:
                    emit_progress(session_id, f"Warning: Layer {layer_name} has constant values. Skipping normalization.", socketio)
                
                if total_suitability is None:
                    total_suitability = np.zeros_like(layer_data)
                
                # Add weighted layer to total suitability
                total_suitability += layer_data * weight
                
            emit_progress(session_id, f"Added {layer_name} to total suitability with weight {weight}", socketio)
        
        # Normalize total suitability to 0-100 range
        if total_weight > 0:
            total_suitability = (total_suitability / total_weight) * 100
        else:
            emit_error(session_id, "Total weight is zero. Cannot normalize suitability scores.", socketio)
            return None
        
        # Add final debug information
        emit_progress(session_id, f"Final suitability stats: min={np.nanmin(total_suitability)}, max={np.nanmax(total_suitability)}, mean={np.nanmean(total_suitability)}, NaN count={np.isnan(total_suitability).sum()}", socketio)
        
        emit_progress(session_id, "Weighted suitability calculation completed", socketio)
        return total_suitability
    
    except Exception as e:
        emit_error(session_id, f"Error in weighted suitability calculation: {str(e)}", socketio)
        return None



def run_full_spatial_operations(engine, session_id, socketio):
    """
    Perform comprehensive landfill site suitability analysis.
    
    Args:
    engine: SQLAlchemy engine for database connection
    session_id (str): Unique identifier for the current session
    socketio: SocketIO instance for real-time communication
    """
    try:
        emit_progress(session_id, "Initiating comprehensive landfill site suitability analysis.", socketio)
        
        # Fetch Nyeri boundary
        nyeri_gdf = fetch_nyeri_boundary(engine, session_id, socketio)
        if nyeri_gdf is None:
            raise ValueError("Nyeri boundary could not be fetched.")
        
    # Calculate weights using AHP
        weights_dict, matrix = calculate_weights()
        emit_progress(session_id, f"AHP Weights calculated: {weights_dict}", socketio)
         
        # Check consistency of AHP matrix
        consistency_ratio, consistency_index, lambda_max = calculate_consistency_ratio(matrix, weights_dict)
        if consistency_ratio is not None:
            emit_progress(session_id, f"Consistency Ratio: {consistency_ratio:.4f}", socketio)
            emit_progress(session_id, f"Consistency Index: {consistency_index:.4f}", socketio)
            emit_progress(session_id, f"Lambda Max: {lambda_max:.4f}", socketio)
    
            if consistency_ratio < 0.1:
               emit_progress(session_id, "AHP matrix is consistent (CR < 0.1).", socketio)
            else:
                emit_progress(session_id, "Warning: The pairwise comparison matrix is not consistent (CR >= 0.1).", socketio)
                emit_progress(session_id, "Please revise the comparison matrix to improve consistency.", socketio)
        else:
            emit_error(session_id, "Could not calculate consistency ratio.", socketio)
                
        # Define feature types and buffer distances
        feature_types = [
            ('River', [300, 1000, 1500, 2000]),
            ('Road', [400, 800, 1000, 1200]),
            ('ProtectedArea', [300, 1000, 1500, 2000, 2500]),
            ('Settlement', [400, 900, 1500, 2100])
        ]
        
        all_buffers = []
        all_original = []
        buffer_images = {}
        layers = {}
        
        # Create common grid for all layers
        bounds = nyeri_gdf.total_bounds
        res = 30  # 30m resolution
        rows = int((bounds[3] - bounds[1]) / res)
        cols = int((bounds[2] - bounds[0]) / res)
        transform = from_bounds(*bounds, cols, rows)
        emit_progress(session_id, f"Created transform with rows={rows}, cols={cols}", socketio)
        
        # Process each feature type
        for feature_type, distances in feature_types:
            emit_progress(session_id, f"Processing {feature_type} data.", socketio)
            
            # Fetch data from PostGIS
            query = f'SELECT id, geom FROM "{feature_type}"'
            gdf = fetch_data_from_postgis(query, engine, session_id, socketio)
            
            if gdf is not None:
                # Reproject and mask data to Nyeri boundary
                gdf_projected = reproject_to_nyeri(gdf, session_id, socketio)
                gdf_masked = mask_to_nyeri(gdf_projected, nyeri_gdf, session_id, socketio)
                
                if 'geometry' in gdf_masked.columns:
                    gdf_masked = gdf_masked.rename(columns={'geom': 'geometry'}).set_geometry('geometry')
                
                geometry_types = gdf_masked.geometry.geom_type.unique()
                emit_progress(session_id, f"Geometry types for {feature_type}: {geometry_types}", socketio)
                
                # Create buffer zones
                buffers = create_arcgis_like_buffers(gdf_masked, distances, session_id, socketio, nyeri_gdf)
                
            if buffers:
             all_buffers.append(buffers)
             all_original.append(gdf_masked)
             
             # Plot buffer zones with session ID in filename
             buffer_plot_path = plot_buffers(
                 buffers, 
                 gdf_masked, 
                 nyeri_gdf, 
                 session_id, 
                 socketio, 
                 title=f'{feature_type} Buffers'
             )
             if buffer_plot_path:
                 buffer_images[feature_type] = buffer_plot_path
             
             # Rasterize buffer zones
             suitability_mapping = globals()[f"{feature_type.lower()}_suitability_mapping"]
             buffer_raster = rasterize_buffers(
                 gdf_masked,
                 buffers,
                 transform,
                 (rows, cols),
                 suitability_mapping,
                 distances,
                 session_id,
                 socketio
             )
             
             # Save buffer raster as TIF with session ID
             buffer_tif_path = os.path.join("output", f'{feature_type.lower()}_buffer_session_{session_id}.tif')
             save_raster_as_tif(
                 buffer_raster, 
                 transform, 
                 NYERI_CRS, 
                 buffer_tif_path, 
                 session_id, 
                 socketio
             )
             
             # Create suitability map visualization with session ID
             suitability_map_path = os.path.join(
                 'output', 
                 f'{feature_type.lower()}_suitability_map_session_{session_id}.png'
             )
             create_suitability_map(
                 buffer_raster,
                 f'{feature_type} Suitability Map',
                 suitability_map_path,
                 transform,
                 NYERI_CRS,
                 nyeri_gdf,
                 plot_boundary=True
             )
             buffer_images[f'{feature_type}Suitability'] = suitability_map_path
             layers[f'Buffer_{feature_type}'] = {'path': buffer_tif_path}
             emit_progress(session_id, f"Rasterized and saved {feature_type} buffer", socketio)
        
        # Create combined buffer zones plot
        if all_buffers and all_original:
            emit_progress(session_id, "Creating combined buffer zones plot.", socketio)
            try:
                concatenated_original = pd.concat(all_original, ignore_index=True)
                # all_buffers_combined = gpd.GeoDataFrame(pd.concat(
                #     [gdf for buffer_list in all_buffers for gdf in buffer_list], ignore_index=True
                # ))
                
                # # Ensure the combined buffers are a GeoDataFrame
                # if not isinstance(all_buffers_combined, gpd.GeoDataFrame):
                #     raise ValueError("Combined buffers are not a GeoDataFrame.")
                
                # # Ensure the concatenated original is a GeoDataFrame
                # if not isinstance(concatenated_original, gpd.GeoDataFrame):
                #     raise ValueError("Concatenated original is not a GeoDataFrame.")
                
                # combined_image_path = plot_buffers(
                #     all_buffers_combined,
                #     concatenated_original,
                #     nyeri_gdf, 
                #     session_id, 
                #     socketio, 
                #     title='Combined Buffer Zones'
                # )
                # if combined_image_path:
                #     buffer_images['Combined'] = combined_image_pat
                # emit_progress(session_id, f"Combined buffer zones plot created: {combined_image_path}", socketio)
            except Exception as e:
                emit_error(session_id, f"Error creating combined buffer zones: {str(e)}", socketio)
        else:
            emit_error(session_id, "No valid data to create combined buffer zones.", socketio)
        
        # Process DEM and slope
        emit_progress(session_id, "Starting DEM and slope analysis.", socketio)
        dem_gdf = fetch_dem_data(engine, session_id, socketio)
        if dem_gdf is not None:
            bounds = dem_gdf.total_bounds
            res = 30  # 30m resolution
            rows = int((bounds[3] - bounds[1]) / res)
            cols = int((bounds[2] - bounds[0]) / res)
            transform = from_bounds(*bounds, cols, rows)
            
            # Rasterize DEM data
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
            
            # Fill NaN values with nearest valid value
            dem_raster_filled = generic_filter(dem_raster, lambda x: np.nanmean(x) if np.isnan(x).all() else np.nanmean(x), size=3, mode='nearest')
            emit_progress(session_id, "NaN values in DEM raster have been filled.", socketio)
            
            # Calculate slope
            emit_progress(session_id, "Starting slope calculation.", socketio)
            slope = calculate_slope(dem_raster_filled, transform, session_id, socketio)

            if slope is not None:
                # Visualize and save the slope
                plt.figure(figsize=(15, 15))
                extent = rasterio.plot.plotting_extent(dem_raster_filled, transform)
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
                save_raster_as_tif(slope, transform, NYERI_CRS, slope_tif_path, session_id, socketio)
                layers['Slope'] = {'path': slope_tif_path}
                
                emit_progress(session_id, f"Slope analysis completed and results saved: {slope_tif_path}", socketio)
            else:
                emit_error(session_id, "Slope calculation failed.", socketio)
        else:
            emit_error(session_id, "DEM data could not be fetched or processed.", socketio)
        
        # Process geology and soil data
        # geology_gdf = fetch_and_classify_vector('Geology', geology_suitability_mapping, engine, session_id, socketio, nyeri_gdf)
        soil_gdf = fetch_and_classify_vector('Soil', soil_suitability_mapping, engine, session_id, socketio, nyeri_gdf)
        
        # Process land use data
        emit_progress(session_id, "Starting land use processing", socketio)
        landuse_raster, landuse_transform, landuse_crs, landuse_plot_path, land_use_types = fetch_and_plot_landuse_raster(engine, session_id, socketio)
        if landuse_raster is not None:
            buffer_images['LandUse'] = landuse_plot_path
            landuse_suitability = process_landuse_suitability(landuse_raster, land_use_types, session_id, socketio)
            
            # Save land use suitability as TIF
            landuse_tif_path = os.path.join("output", f"landuse_suitability_session_{session_id}.tif")
            save_raster_as_tif(landuse_suitability, landuse_transform, landuse_crs, landuse_tif_path, session_id, socketio)
            layers['LandUse'] = {'path': landuse_tif_path}
            
            emit_progress(session_id, "Land use suitability processing completed", socketio)
        else:
            emit_error(session_id, "Failed to fetch land use raster.", socketio)
            landuse_suitability = None
        
        # Create common grid for all layers
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

            # Rasterize vector layers to common grid
            emit_progress(session_id, "Rasterizing vector layers to common grid...", socketio)
            for i, (feature_type, distances) in enumerate(feature_types):
                try:
                    suitability_mapping = globals()[f"{feature_type.lower()}_suitability_mapping"]
                    buffer_raster = rasterize_buffers(
                        all_original[i],
                        all_buffers[i],
                        transform,
                        (rows, cols),
                        suitability_mapping,
                        distances,
                        session_id,
                        socketio
                    )
                    tif_path = os.path.join("output", f'{feature_type.lower()}_buffer.tif')
                    save_raster_as_tif(buffer_raster, transform, NYERI_CRS, tif_path, session_id, socketio)
                    layers[f'Buffer_{feature_type}'] = {'path': tif_path}
                    emit_progress(session_id, f"Rasterized {feature_type} buffer", socketio)
                except Exception as e:
                    emit_error(session_id, f"Error rasterizing {feature_type} buffer: {str(e)}", socketio)

            # # Rasterize geology and soil layers
            # if geology_gdf is not None:
            #     try:
            #         geology_raster = rasterize(
            #             [(geom, value) for geom, value in zip(geology_gdf.geometry, geology_gdf.suitability_score)],
            #             out_shape=(rows, cols),
            #             transform=transform,
            #             fill=np.nan,
            #             dtype='float32'
            #         )
            #         geology_tif_path = os.path.join("output", 'geology.tif')
            #         save_raster_as_tif(geology_raster, transform, NYERI_CRS, geology_tif_path, session_id, socketio)
            #         layers['Geology'] = {'path': geology_tif_path}
            #         emit_progress(session_id, "Rasterized Geology layer", socketio)
            #     except Exception as e:
            #         emit_error(session_id, f"Error rasterizing Geology layer: {str(e)}", socketio)

            if soil_gdf is not None:
                try:
                    soil_raster = rasterize(
                        [(geom, value) for geom, value in zip(soil_gdf.geometry, soil_gdf.suitability_score)],
                        out_shape=(rows, cols),
                        transform=transform,
                        fill=np.nan,
                        dtype='float32'
                    )
                    soil_tif_path = os.path.join("output", f"soil_suitability_session_{session_id}.tif")
                    save_raster_as_tif(soil_raster, transform, NYERI_CRS, soil_tif_path, session_id, socketio)
                    layers['Soil'] = {'path': soil_tif_path}
                    emit_progress(session_id, "Rasterized Soil layer", socketio)
                except Exception as e:
                    emit_error(session_id, f"Error rasterizing Soil layer: {str(e)}", socketio)

            # Resample land use and slope layers
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
                    landuse_tif_path = os.path.join("output", 'landuse.tif')
                    save_raster_as_tif(resampled_landuse, transform, NYERI_CRS, landuse_tif_path, session_id, socketio)
                    layers['LandUse'] = {'path': landuse_tif_path}
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
                    slope_suitability = np.vectorize(slope_suitability_mapping)(resampled_slope)
                    slope_tif_path = os.path.join("output", f"slope_suitability_session_{session_id}.tif")
                    save_raster_as_tif(slope_suitability, transform, NYERI_CRS, slope_tif_path, session_id, socketio)
                    layers['Slope'] = {'path': slope_tif_path}
                    emit_progress(session_id, "Slope layer resampled and added to common grid", socketio)
                except Exception as e:
                    emit_error(session_id, f"Error adding Slope layer: {str(e)}", socketio)

            # Ensure all paths in the layers dictionary exist
            for layer_name, layer_info in layers.items():
                if not os.path.exists(layer_info['path']):
                    raise FileNotFoundError(f"Raster file for {layer_name} not found at {layer_info['path']}")

            # Calculate weighted suitability
            emit_progress(session_id, "Starting weighted suitability calculation", socketio)
            total_suitability = calculate_weighted_suitability(layers, weights_dict, session_id, socketio)

            if total_suitability is not None:
                # Reclassify suitability scores
                reclassified_suitability = reclassify_suitability(total_suitability)
                # After processing all layers but before grid analysis
              
                try:
                    # Prepare raster criteria using the already processed paths
                    raster_criteria = {
                        'Slope': layers['Slope']['path'],
                        'Land_Use': layers['LandUse']['path'],
                        'Soil': layers['Soil']['path']
                    }
                    
                    # Verify raster files exist
                    for name, path in raster_criteria.items():
                        if not os.path.exists(path):
                            raise FileNotFoundError(f"Missing raster file for {name}: {path}")
                    
                    # Create buffer sets dictionary using the already processed buffers
                    buffer_sets = {}
                    for i, (feature_type, _) in enumerate(feature_types):
                        if i < len(all_buffers):
                            clean_name = feature_type.replace('ProtectedArea', 'Protected_Areas')
                            buffer_sets[clean_name] = all_buffers[i]
                            
                               # Create suitability maps for each layer
                    for layer_name, layer_info in layers.items():
                      clean_layer_name = layer_name.replace('Buffer_', '')
                      output_path = os.path.join('output', f'{layer_name.lower()}_suitability_map_session_{session_id}.png')
                    with rasterio.open(layer_info['path']) as src:
                        layer_data = src.read(1)
                        plot_boundary = layer_name.lower() not in ['soil', 'landuse', 'slope', 'geology']
                        create_suitability_map(layer_data, f'{clean_layer_name} Suitability Map', output_path, transform, NYERI_CRS, nyeri_gdf, plot_boundary=plot_boundary)
                    buffer_images[f'{layer_name}Suitability'] = output_path
                    emit_progress(session_id, f"Created suitability map for {clean_layer_name}", socketio)
                    
                           
                     # Create training dataset
                    emit_progress(session_id, "Creating training dataset...", socketio)
                    training_gdf, plot_path, csv_path, model_results = create_training_dataset(
                        nyeri_gdf=nyeri_gdf,
                        buffer_sets=buffer_sets,
                        raster_criteria=raster_criteria,
                        n_points=10000,
                        session_id=session_id,
                        socketio=socketio
                    )
                    
                    if training_gdf is not None:
                        # Store original training dataset paths
                        buffer_images['Training_Points'] = plot_path.replace('\\', '/')
                        buffer_images['Training_Data_CSV'] = csv_path.replace('\\', '/')
                        
                        # Store model-related paths and make predictions
                        if model_results:
                            # Store all visualization paths
                            if 'visualizations' in model_results:
                                for viz_name, viz_path in model_results['visualizations'].items():
                                    buffer_images[f'Model_{viz_name}'] = viz_path.replace('\\', '/')
                            
                            # Add prediction for entire map
                            output_dir = os.path.join(os.getcwd(), 'output')
                            output_path = os.path.join(output_dir, f'suitability_prediction_{session_id}.tif')
                            
                            model_path = os.path.join('output', f'model_{session_id}.joblib')
                            scaler_path = os.path.join('output', f'scaler_{session_id}.joblib')
                            
                        #     raster_criteria = {
                        #     'Slope': os.path.join('output', f'slope_suitability_session_{session_id}.tif'),
                        #     'Land_Use': os.path.join('output', f'landuse_suitability_session_{session_id}.tif'),
                        #     'Soil': os.path.join('output', f'soil_suitability_session_{session_id}.tif')
                        #      }

                        # # Prepare buffer sets with the correct filenames
                        #     buffer_setss = {
                        #     'River': os.path.join('output', f'river_buffer_session_{session_id}.tif'),
                        #     'Road': os.path.join('output', f'road_buffer_session_{session_id}.tif'),
                        #     'Settlement': os.path.join('output', f'settlement_buffer_session_{session_id}.tif'),
                        #     'Protected_Areas': os.path.join('output', f'protectedarea_buffer_session_{session_id}.tif')
                        #        }
                
                            
                    #         prediction_results = predict_map_suitability(
                    #             nyeri_gdf=nyeri_gdf,
                    #             raster_criteria=raster_criteria,
                    #             buffer_sets=buffer_sets,
                    #             model_path=model_path,
                    #             scaler_path=scaler_path,
                    #             interval=25,
                    #             session_id=session_id,
                    #             socketio=socketio,
                    #             engine=engine
                    #         )
                            
                    #         if prediction_results:
                    #             # Access the statistics from the results dictionary
                    #             stats = prediction_results['stats']
                                
                    #             # Update buffer images with all the paths
                    #             buffer_images['Suitability_Prediction'] = prediction_results['full_map_path'].replace('\\', '/')
                    #             buffer_images['Candidate_Sites'] = prediction_results['candidate_map_path'].replace('\\', '/')
                    #             if prediction_results['continuous_map_path']:
                    #                 buffer_images['Continuous_Sites'] = prediction_results['continuous_map_path'].replace('\\', '/')
                                
                    #             emit_progress(session_id, "✅ Map-wide suitability prediction completed successfully.", socketio)
                    #             emit_progress(session_id, "\n📊 Prediction Statistics:", socketio)
                                
                    #             # Full map statistics
                    #             emit_progress(session_id, "\nFull Map Statistics:", socketio)
                    #             emit_progress(session_id, f"• Total points analyzed: {stats['full_map']['total_points']}", socketio)
                    #             emit_progress(session_id, f"• Minimum suitability: {stats['full_map']['min_score']:.2f}", socketio)
                    #             emit_progress(session_id, f"• Maximum suitability: {stats['full_map']['max_score']:.2f}", socketio)
                    #             emit_progress(session_id, f"• Mean suitability: {stats['full_map']['mean_score']:.2f}", socketio)
                                
                    #             # Class distribution
                    #             emit_progress(session_id, "\n📊 Area Distribution:", socketio)
                    #             for class_name, count in stats['full_map']['class_distribution'].items():
                    #                 percentage = (count / stats['full_map']['total_points']) * 100
                    #                 emit_progress(session_id, f"• {class_name}: {count} points ({percentage:.1f}%)", socketio)
                                
                    #             # Candidate sites statistics
                    #             if 'continuous_sites' in stats:
                    #                 emit_progress(session_id, "\n📊 Continuous Sites Statistics:", socketio)
                    #                 emit_progress(session_id, f"• Total suitable sites: {stats['continuous_sites']['total_sites']}", socketio)
                    #                 emit_progress(session_id, f"• Total area: {stats['continuous_sites']['total_area_ha']:.2f} ha", socketio)
                    #                 emit_progress(session_id, f"• Minimum site area: {stats['continuous_sites']['min_area_ha']:.2f} ha", socketio)
                    #                 emit_progress(session_id, f"• Maximum site area: {stats['continuous_sites']['max_area_ha']:.2f} ha", socketio)
                    #                 emit_progress(session_id, f"• Mean site area: {stats['continuous_sites']['mean_area_ha']:.2f} ha", socketio)
                    #         else:
                    #             emit_error(session_id, "❌ Failed to create map-wide suitability prediction.", socketio)
                                                        
                    #         emit_progress(session_id, "Training dataset and model created successfully.", socketio)
                    #     else:
                    #         emit_error(session_id, "Training dataset created but model training failed.", socketio)
                    # else:
                    #     emit_error(session_id, "Failed to create training dataset.", socketio)
                                    
                except Exception as e:
                            emit_error(session_id, f"Error creating training dataset: {str(e)}", socketio)
                            traceback.print_exc()
                                                    
 

                # # # Create total suitability map
                # # output_path = os.path.join('output', f'total_suitability_map_session_{session_id}.png')
                # # create_suitability_map(reclassified_suitability, 'Total Suitability Map', output_path, transform, NYERI_CRS, nyeri_gdf, plot_boundary=True)
                # # buffer_images['TotalSuitability'] = output_path
                # # emit_progress(session_id, "Created total suitability map", socketio)

                # # suitability_raster_path = os.path.join("output", f'total_suitability_map_session_{session_id}.tif')
                # # save_raster_as_tif(total_suitability, transform, NYERI_CRS, suitability_raster_path, session_id, socketio)
                # # layers['TotalSuitability'] = {'path': suitability_raster_path}
                # # emit_progress(session_id, f"Total suitability raster saved as {suitability_raster_path}", socketio)
                
                #  waste collection
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
                    emit_progress(session_id, "Waste collection optimization complested but found no optimal locations.", socketio)
                    
                    #  network anlaysis
                try:
                   emit_progress(session_id, "Starting network analysis with dummy data...", socketio)
                   network_results = perform_network_analysis(
                             nyeri_gdf=nyeri_gdf,
                             session_id=session_id,
                             socketio=socketio,
                             engine=engine
                        )
                         
                   if network_results:
                             buffer_images['NetworkAnalysis'] = network_results['static_map'].replace('\\', '/')
                             emit_progress(session_id, "Network analysis completed successfully.", socketio)
                except Exception as e:
                    emit_error(session_id, f"Error in network analysis: {str(e)}", socketio)
                  
  
                for key, path in buffer_images.items():
                    if isinstance(path, str):
                        buffer_images[key] = path.replace('\\', '/')
                    elif isinstance(path, dict):
                        for nested_key, nested_path in path.items():
                            path[nested_key] = nested_path.replace('\\', '/')

                socketio.emit('buffer_images', {'session_id': session_id, 'images': buffer_images}, room=session_id)

                emit_progress(session_id, "All spatial operations and ML predictions completed successfully.", socketio)
                socketio.emit('operation_completed', {'session_id': session_id, 'message': 'Operations completed successfully.'}, room=session_id)
            else:
                emit_error(session_id, "Failed to calculate total suitability", socketio)
        else:
            emit_error(session_id, "No valid data to create suitability map.", socketio)

    except Exception as e:
        emit_error(session_id, f"Unexpected error during spatial operations: {str(e)}", socketio)
        emit_error(session_id, f"Traceback: {traceback.format_exc()}", socketio)
    finally:
        pass
