import os
import sys
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patheffects as pe
from shapely.geometry import box, Point
from sklearn.cluster import KMeans
from rasterio.plot import plotting_extent
from rasterio.mask import mask
import rasterio
import rasterio.transform
from flask import Flask
from flask_socketio import SocketIO
from sqlalchemy import create_engine
import logging
from pyproj import CRS

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define Coordinate Reference System
AOI_CRS = CRS.from_epsg(21037)  # Arc 1960 / UTM zone 37S

app = Flask(__name__)
socketio = SocketIO(app)

def emit_progress(message, socketio, session_id):
    logger.info(f"Progress: {message}")
    if socketio and session_id:
        socketio.emit('progress_update', {'session_id': session_id, 'message': message}, room=session_id)

def reproject_to_aoi_crs(gdf, socketio=None, session_id=None):
    emit_progress(f"Reprojecting data to {AOI_CRS}", socketio, session_id)
    try:
        if gdf.crs is None:
            emit_progress("Warning: Input GeoDataFrame has no CRS. Assuming it's already in the correct CRS.", socketio, session_id)
            gdf.set_crs(AOI_CRS, inplace=True)
        elif gdf.crs != AOI_CRS:
            gdf = gdf.to_crs(AOI_CRS)
        return gdf
    except Exception as e:
        emit_progress(f"Error during reprojection: {str(e)}", socketio, session_id)
        logger.error(f"Error during reprojection: {str(e)}")
        return None

def fetch_data_from_db(engine, query, socketio, session_id):
    emit_progress(f"Fetching data with query: {query}", socketio, session_id)
    try:
        gdf = gpd.read_postgis(query, engine, geom_col='geom')
        if gdf.empty:
            emit_progress("Fetched GeoDataFrame is empty.", socketio, session_id)
            return None
        gdf = reproject_to_aoi_crs(gdf, socketio, session_id)
        return gdf
    except Exception as e:
        emit_progress(f"Error fetching data: {str(e)}", socketio, session_id)
        logger.error(f"Error fetching data: {str(e)}")
        return None

def validate_geometries(gdf, socketio, session_id):
    emit_progress("Validating geometries.", socketio, session_id)
    invalid = ~gdf.is_valid
    if invalid.any():
        emit_progress(f"Found {invalid.sum()} invalid geometries. Attempting to fix.", socketio, session_id)
        gdf.loc[invalid, 'geometry'] = gdf.loc[invalid, 'geometry'].buffer(0)
        emit_progress("Invalid geometries fixed.", socketio, session_id)
    else:
        emit_progress("All geometries are valid.", socketio, session_id)
    return gdf

def create_grid(aoi_gdf, cell_size=100, socketio=None, session_id=None):
    emit_progress(f"Creating regular grid with cell size {cell_size} meters", socketio, session_id)
    
    bounds = aoi_gdf.total_bounds
    xmin, ymin, xmax, ymax = bounds
    x_coords = np.arange(xmin, xmax + cell_size, cell_size)
    y_coords = np.arange(ymin, ymax + cell_size, cell_size)
    
    grid_polygons = [box(x, y, x + cell_size, y + cell_size) for x in x_coords[:-1] for y in y_coords[:-1]]
    
    grid_gdf = gpd.GeoDataFrame({'geometry': grid_polygons}, crs=aoi_gdf.crs)
    
    grid_gdf = validate_geometries(grid_gdf, socketio, session_id)
    
    grid_gdf['is_within_aoi'] = grid_gdf.within(aoi_gdf.unary_union)
    
    grid_gdf['x_idx'] = ((grid_gdf.geometry.bounds['minx'] - xmin) // cell_size).astype(int)
    grid_gdf['y_idx'] = ((grid_gdf.geometry.bounds['miny'] - ymin) // cell_size).astype(int)
    
    emit_progress(f"Total grid cells created: {len(grid_gdf)}", socketio, session_id)
    emit_progress(f"Grid cells within AOI: {grid_gdf['is_within_aoi'].sum()}", socketio, session_id)
    
    return grid_gdf, x_coords, y_coords

def calculate_proximity(grid_gdf, roads_gdf, settlements_gdf, socketio=None, session_id=None):
    emit_progress("Calculating proximity to roads and settlements", socketio, session_id)
    
    emit_progress("Validating geometries of roads and settlements.", socketio, session_id)
    roads_gdf = validate_geometries(roads_gdf, socketio, session_id)
    settlements_gdf = validate_geometries(settlements_gdf, socketio, session_id)
    
    emit_progress("Finding nearest roads for each grid cell", socketio, session_id)
    grid_with_roads = gpd.sjoin_nearest(grid_gdf, roads_gdf, how="left", distance_col="road_distance")
    
    emit_progress("Finding nearest settlements for each grid cell", socketio, session_id)
    grid_with_settlements = gpd.sjoin_nearest(
        grid_with_roads, settlements_gdf,
        how="left", distance_col="settlement_distance",
        lsuffix='grid', rsuffix='settlement'
    )
    
    grid_with_settlements['road_distance'] = grid_with_settlements['road_distance'].fillna(np.inf)
    grid_with_settlements['settlement_distance'] = grid_with_settlements['settlement_distance'].fillna(np.inf)
    
    emit_progress("Calculating suitability scores", socketio, session_id)
    
    grid_with_settlements['road_score'] = grid_with_settlements['road_distance'].apply(
        lambda x: 5 if x < 50 else (4 if x < 200 else (3 if x < 500 else (2 if x < 1000 else 1)))
    )
    
    grid_with_settlements['settlement_score'] = grid_with_settlements['settlement_distance'].apply(
        lambda x: 5 if x < 100 else (4 if x < 500 else (3 if x < 1000 else (2 if x < 2000 else 1)))
    )
    
    grid_with_settlements['combined_score'] = grid_with_settlements[['road_score', 'settlement_score']].min(axis=1)
    
    grid_gdf = grid_gdf.merge(
        grid_with_settlements[['road_distance', 'settlement_distance', 'road_score', 'settlement_score', 'combined_score']],
        left_index=True,
        right_index=True,
        how='left'
    )
    
    return grid_gdf

def perform_clustering(grid_gdf, n_clusters=5, socketio=None, session_id=None):
    emit_progress(f"Performing clustering with {n_clusters} clusters", socketio, session_id)
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        required_columns = ['road_distance', 'settlement_distance']
        missing_columns = [col for col in required_columns if col not in grid_gdf.columns]
        if missing_columns:
            emit_progress(f"Missing columns for clustering: {missing_columns}", socketio, session_id)
            logger.error(f"Missing columns for clustering: {missing_columns}")
            return grid_gdf
        
        grid_gdf[['road_distance', 'settlement_distance']] = grid_gdf[['road_distance', 'settlement_distance']].fillna(0)
        features = grid_gdf[['road_distance', 'settlement_distance']].values
        grid_gdf['cluster'] = kmeans.fit_predict(features)
        emit_progress("Clustering completed successfully.", socketio, session_id)
        return grid_gdf
    except Exception as e:
        emit_progress(f"Error during clustering: {str(e)}", socketio, session_id)
        logger.error(f"Error during clustering: {str(e)}")
        return grid_gdf

def select_optimal_locations(grid_gdf, socketio=None, session_id=None):
    emit_progress("Selecting optimal locations", socketio, session_id)
    try:
        optimal = grid_gdf.loc[grid_gdf.groupby('cluster')['combined_score'].idxmax()]
        emit_progress(f"Selected {len(optimal)} optimal locations.", socketio, session_id)
        return optimal
    except Exception as e:
        emit_progress(f"Error selecting optimal locations: {str(e)}", socketio, session_id)
        logger.error(f"Error selecting optimal locations: {str(e)}")
        return pd.DataFrame()

def mask_raster_with_boundary(raster_data, transform, aoi_gdf, socketio=None, session_id=None):
    emit_progress("Masking raster data with Area of Interest boundary", socketio, session_id)
    try:
        aoi_gdf = validate_geometries(aoi_gdf, socketio, session_id)
        
        aoi_geojson = [aoi_gdf.geometry.unary_union.__geo_interface__]
        
        output_dir = os.path.dirname("output/temp_masked.tif")
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(
                driver='GTiff',
                height=raster_data.shape[0],
                width=raster_data.shape[1],
                count=1,
                dtype=raster_data.dtype,
                crs=AOI_CRS,
                transform=transform
            ) as dataset:
                dataset.write(raster_data, 1)
                out_image, out_transform = mask(dataset, aoi_geojson, crop=True, nodata=np.nan)
        return out_image[0], out_transform
    except rasterio.errors.CRSError as e:
        emit_progress(f"CRS Error in mask_raster_with_boundary: {str(e)}", socketio, session_id)
        logger.error(f"CRS Error in mask_raster_with_boundary: {str(e)}")
    except Exception as e:
        emit_progress(f"Error in mask_raster_with_boundary: {str(e)}", socketio, session_id)
        logger.error(f"Error in mask_raster_with_boundary: {str(e)}")
    return None, None

def create_suitability_map(data, title, output_path, transform, aoi_gdf, socketio=None, session_id=None):
    emit_progress(f"Creating suitability map: {title}", socketio, session_id)
    
    if transform is None:
        emit_progress(f"Error: Transform is None for {title}. Skipping map creation.", socketio, session_id)
        logger.error(f"Transform is None for {title}.")
        return
    
    masked = mask_raster_with_boundary(data, transform, aoi_gdf, socketio, session_id)
    if masked[0] is None:
        emit_progress(f"Error masking raster data for {title}. Skipping map creation.", socketio, session_id)
        logger.error(f"Error masking raster data for {title}.")
        return
    
    masked_data, masked_transform = masked
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    cmap = colors.ListedColormap(['purple', 'red', 'yellow', 'lightgreen', 'darkgreen'])
    norm = colors.BoundaryNorm([1, 2, 3, 4, 5, 6], cmap.N)
    
    cmap.set_bad(color='white', alpha=0)
    
    extent = plotting_extent(masked_data, masked_transform)
    im = ax.imshow(masked_data, cmap=cmap, norm=norm, extent=extent)
    
    aoi_gdf.boundary.plot(ax=ax, edgecolor='black', linewidth=1, label='Area of Interest Boundary')
    
    legend_elements = [
        mpatches.Patch(color='purple', label='Not suitable'),
        mpatches.Patch(color='red', label='Less suitable'),
        mpatches.Patch(color='yellow', label='Moderately suitable'),
        mpatches.Patch(color='lightgreen', label='Suitable'),
        mpatches.Patch(color='darkgreen', label='Highly suitable')
    ]
    legend = ax.legend(handles=legend_elements, loc='lower left', fontsize=8, bbox_to_anchor=(0, -0.3))
    
    map_info = ax.text(0.5, -0.3, 'Coordinate system: Arc 1960 UTM Zone 37S\nProjection: Transverse Mercator\nDatum: Arc 1960',
                       transform=ax.transAxes, fontsize=8, ha='right', va='top')
    
    ax.set_title(f'{title}', fontsize=16)
    ax.set_xlabel('Easting (meters)')
    ax.set_ylabel('Northing (meters)')
    
    scalebar = ScaleBar(1, location='lower center', scale_loc='bottom', length_fraction=0.5, units='km', dimension='si-length', label='Scale')
    ax.add_artist(scalebar)
    
    ax.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction', 
                horizontalalignment='center', verticalalignment='center',
                fontsize=16, fontweight='bold', path_effects=[pe.withStroke(linewidth=3, foreground="w")])
    ax.arrow(0.95, 0.93, 0, 0.02, head_width=0.01, head_length=0.01, 
             fc='k', ec='k', transform=ax.transAxes)
    
    fig.patch.set_linewidth(2)
    fig.patch.set_edgecolor('black')
    
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}'))
    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.4)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    emit_progress(f"Suitability map saved: {output_path}", socketio, session_id)
    logger.info(f"Suitability map saved: {output_path}")

def optimize_waste_collection(engine, session_id, socketio, aoi_gdf):
    emit_progress("Starting waste collection optimization process", socketio, session_id)
    
    roads_query = 'SELECT geom FROM "Road"'
    settlements_query = 'SELECT geom FROM "Settlement"'
    
    roads_gdf = fetch_data_from_db(engine, roads_query, socketio, session_id)
    if roads_gdf is None:
        emit_progress("Failed to fetch road data. Aborting optimization process.", socketio, session_id)
        return None
    
    emit_progress("Simplifying road geometries", socketio, session_id)
    roads_gdf['geometry'] = roads_gdf.geometry.simplify(tolerance=10)
    
    settlements_gdf = fetch_data_from_db(engine, settlements_query, socketio, session_id)
    if settlements_gdf is None:
        emit_progress("Failed to fetch settlement data. Aborting optimization process.", socketio, session_id)
        return None
    
    emit_progress("Simplifying settlement geometries", socketio, session_id)
    settlements_gdf['geometry'] = settlements_gdf.geometry.simplify(tolerance=10)
    
    aoi_boundary = reproject_to_aoi_crs(aoi_gdf, socketio, session_id)
    if aoi_boundary is None or aoi_boundary.empty:
        emit_progress("Area of Interest boundary data is missing or empty. Aborting optimization process.", socketio, session_id)
        return None
    
    cell_size = 100  # Define cell size in meters
    grid_gdf, x_coords, y_coords = create_grid(aoi_boundary, cell_size=cell_size, socketio=socketio, session_id=session_id)
    
    if grid_gdf is None or grid_gdf.empty:
        emit_progress("Grid creation failed. Aborting optimization process.", socketio, session_id)
        return None
    
    grid_gdf = calculate_proximity(grid_gdf, roads_gdf, settlements_gdf, socketio, session_id)
    
    grid_side_x = len(x_coords)
    grid_side_y = len(y_coords)
    actual_points = len(grid_gdf)
    
    emit_progress(f"Grid dimensions: {grid_side_x} (x) x {grid_side_y} (y)", socketio, session_id)
    emit_progress(f"Total grid points: {actual_points}", socketio, session_id)
    emit_progress(f"Grid points within AOI: {grid_gdf['is_within_aoi'].sum()}", socketio, session_id)
    
    try:
        road_suitability = np.full((grid_side_y, grid_side_x), np.nan)
        settlement_suitability = np.full((grid_side_y, grid_side_x), np.nan)
        combined_suitability = np.full((grid_side_y, grid_side_x), np.nan)
        
        road_suitability[grid_gdf['y_idx'], grid_gdf['x_idx']] = grid_gdf['road_score']
        settlement_suitability[grid_gdf['y_idx'], grid_gdf['x_idx']] = grid_gdf['settlement_score']
        combined_suitability[grid_gdf['y_idx'], grid_gdf['x_idx']] = grid_gdf['combined_score']
        
    except Exception as e:
        emit_progress(f"Error populating suitability arrays: {str(e)}", socketio, session_id)
        logger.error(f"Error populating suitability arrays: {str(e)}")
        return None
    
    transform = rasterio.transform.from_origin(x_coords[0], y_coords[-1], cell_size, cell_size)
    create_suitability_map(road_suitability, "Road Suitability", "output/road_suitability.png", transform, aoi_boundary, socketio, session_id)
    create_suitability_map(settlement_suitability, "Settlement Suitability", "output/settlement_suitability.png", transform, aoi_boundary, socketio, session_id)
    create_suitability_map(combined_suitability, "Combined Suitability", "output/combined_suitability.png", transform, aoi_boundary, socketio, session_id)
    
    grid_gdf = perform_clustering(grid_gdf, n_clusters=5, socketio=socketio, session_id=session_id)
    if grid_gdf is None:
        emit_progress("Clustering failed. Aborting optimization process.", socketio, session_id)
        return None
    
    optimal_locations = select_optimal_locations(grid_gdf, socketio, session_id)
    
    emit_progress("Creating final map with optimal locations", socketio, session_id)
    try:
        fig, ax = plt.subplots(figsize=(20, 20))
        
        aoi_boundary.plot(ax=ax, color='none', edgecolor='black', linewidth=1, zorder=1)
        
        scatter = ax.scatter(
            grid_gdf.geometry.centroid.x, grid_gdf.geometry.centroid.y,
            c=grid_gdf['combined_score'], cmap='RdYlGn',
            s=5, alpha=0.6, zorder=2
        )
        
        roads_gdf.plot(ax=ax, color='gray', linewidth=0.5, zorder=3)
        settlements_gdf.plot(ax=ax, color='blue', markersize=10, alpha=0.7, zorder=4)
        
        if not optimal_locations.empty:
            optimal_locations.plot(ax=ax, color='red', markersize=50, alpha=0.7, zorder=5)
        
        cbar = plt.colorbar(scatter, ax=ax, label='Suitability Score')
        
        ax.set_title('Optimal Waste Collection Points', fontsize=16)
        ax.set_xlabel('Easting (meters)')
        ax.set_ylabel('Northing (meters)')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        ax.legend(
            ['Area of Interest Boundary', 'Grid Points', 'Roads', 'Settlements', 'Optimal Locations'],
            loc='upper left', bbox_to_anchor=(1, 1)
        )
        
        plt.tight_layout()
        final_output_path = 'output/optimal_locations_detailed.png'
        plt.savefig(final_output_path, dpi=300, bbox_inches='tight')
        plt.close()
        emit_progress(f"Final detailed map saved: {final_output_path}", socketio, session_id)
        logger.info(f"Final detailed map saved: {final_output_path}")
        
    except Exception as e:
        emit_progress(f"Error creating final map: {str(e)}", socketio, session_id)
        logger.error(f"Error creating final map: {str(e)}")
        return None
    
    emit_progress("Waste collection optimization process completed", socketio, session_id)
    return optimal_locations, {
        'road_suitability': 'output/road_suitability.png',
        'settlement_suitability': 'output/settlement_suitability.png',
        'combined_suitability': 'output/combined_suitability.png',
        'optimal_locations': 'output/optimal_locations_detailed.png'
    }

@socketio.on('start_optimization')
def handle_optimization(data):
    session_id = data.get('session_id')
    if not session_id:
        emit_progress("No session_id provided.", socketio, None)
        socketio.emit('optimization_failed', {'message': 'No session_id provided.'})
        return
    
    emit_progress("Received optimization request.", socketio, session_id)
    try:
        engine = create_engine('postgresql://username:password@host:port/database')  # Update with your credentials
    except Exception as e:
        emit_progress(f"Database connection failed: {str(e)}", socketio, session_id)
        socketio.emit('optimization_failed', {'message': 'Database connection failed.'}, room=session_id)
        logger.error(f"Database connection failed: {str(e)}")
        return
    
    aoi_query = 'SELECT geom FROM "areaOfInterest"'  # Adjust this query as needed
    aoi_gdf = fetch_data_from_db(engine, aoi_query, socketio, session_id)
    
    if aoi_gdf is None or aoi_gdf.empty:
        socketio.emit('optimization_failed', {'message': 'Failed to load Area of Interest data.'}, room=session_id)
        return
    
    result = optimize_waste_collection(engine, session_id, socketio, aoi_gdf)
    
    if result is None:
        socketio.emit('optimization_failed', {'message': 'Optimization process failed.'}, room=session_id)
        return
    
    optimal_locations, plot_paths = result
    
    if optimal_locations is not None and not optimal_locations.empty:
        emit_progress(f"Waste collection optimization completed. Found {len(optimal_locations)} optimal locations.", socketio, session_id)
        socketio.emit('optimization_complete', {
            'optimal_locations': optimal_locations.to_json(),
            'plot_paths': plot_paths
        }, room=session_id)
    else:
        emit_progress("Optimization completed but no optimal locations were found.", socketio, session_id)
        socketio.emit('optimization_complete', {
            'optimal_locations': None,
            'plot_paths': plot_paths
        }, room=session_id)

if __name__ == "__main__":
    socketio.run(app, debug=True)