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
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
from scipy.ndimage import gaussian_filter, zoom

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define Coordinate Reference System
AOI_CRS = CRS.from_epsg(21037)  # Arc 1960 / UTM zone 37S

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

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

def create_grid(aoi_gdf, cell_size=50, socketio=None, session_id=None):
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

def calculate_proximity(grid_gdf, roads_gdf, settlements_gdf, aoi_gdf, socketio=None, session_id=None):
    emit_progress("Calculating proximity to roads and settlements", socketio, session_id)
    
    emit_progress("Clipping grid to AOI", socketio, session_id)
    grid_gdf = gpd.clip(grid_gdf, aoi_gdf)
    
    emit_progress("Validating geometries of roads and settlements.", socketio, session_id)
    roads_gdf = validate_geometries(roads_gdf, socketio, session_id)
    settlements_gdf = validate_geometries(settlements_gdf, socketio, session_id)
    
    emit_progress("Creating road suitability buffers", socketio, session_id)
    road_buffers = [
        roads_gdf.buffer(50).unary_union.difference(roads_gdf.buffer(0).unary_union),
        roads_gdf.buffer(200).unary_union.difference(roads_gdf.buffer(50).unary_union),
        roads_gdf.buffer(500).unary_union.difference(roads_gdf.buffer(200).unary_union),
        roads_gdf.buffer(1000).unary_union.difference(roads_gdf.buffer(500).unary_union),
        aoi_gdf.geometry.unary_union.difference(roads_gdf.buffer(1000).unary_union)
    ]
    
    road_suitability = gpd.GeoDataFrame(geometry=[geom for geom in road_buffers if not geom.is_empty], crs=aoi_gdf.crs)
    road_suitability['road_score'] = [5, 4, 3, 2, 1][:len(road_suitability)]
    
    emit_progress("Joining road suitability to grid", socketio, session_id)
    grid_gdf = gpd.sjoin(grid_gdf, road_suitability, how="left", predicate="intersects")
    grid_gdf['road_score'] = grid_gdf['road_score'].fillna(1)
    
    emit_progress("Finding nearest settlements for each grid cell", socketio, session_id)
    grid_with_settlements = gpd.sjoin_nearest(
        grid_gdf, settlements_gdf,
        how="left", distance_col="settlement_distance",
        lsuffix='grid', rsuffix='settlement'
    )
    
    grid_with_settlements['settlement_distance'] = grid_with_settlements['settlement_distance'].fillna(np.inf)
    
    emit_progress("Calculating settlement suitability scores", socketio, session_id)
    
    # Settlement score (weight: 0.6)
    grid_with_settlements['settlement_score'] = grid_with_settlements['settlement_distance'].apply(
        lambda x: 5 if x < 100 else (4 if x < 300 else (3 if x < 600 else (2 if x < 1000 else 1)))
    )
    
    # Combined weighted score
    grid_with_settlements['combined_score'] = (
        0.4 * grid_with_settlements['road_score'] + 
        0.6 * grid_with_settlements['settlement_score']
    )
    
    # Normalize combined score to 1-5 range
    min_score = grid_with_settlements['combined_score'].min()
    max_score = grid_with_settlements['combined_score'].max()
    grid_with_settlements['combined_score'] = (
        (grid_with_settlements['combined_score'] - min_score) / (max_score - min_score) * 4 + 1
    )
    
    return grid_with_settlements

def perform_clustering(grid_gdf, n_clusters=5, socketio=None, session_id=None):
    emit_progress(f"Performing clustering with {n_clusters} clusters", socketio, session_id)
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        required_columns = ['road_score', 'settlement_score']
        missing_columns = [col for col in required_columns if col not in grid_gdf.columns]
        if missing_columns:
            emit_progress(f"Missing columns for clustering: {missing_columns}", socketio, session_id)
            logger.error(f"Missing columns for clustering: {missing_columns}")
            return grid_gdf
        
        features = grid_gdf[['road_score', 'settlement_score']].values
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

def create_suitability_map(grid_gdf, score_column, title, output_path, aoi_gdf, socketio=None, session_id=None):
    emit_progress(f"Creating suitability map: {title}", socketio, session_id)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    cmap = colors.LinearSegmentedColormap.from_list("custom", 
        ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641'], N=256)
    norm = colors.BoundaryNorm([1, 2, 3, 4, 5, 6], cmap.N)
    
    grid_gdf.plot(column=score_column, ax=ax, cmap=cmap, norm=norm, legend=False)
    
    aoi_gdf.boundary.plot(ax=ax, edgecolor='black', linewidth=1, label='Area of Interest Boundary')
    
    legend_elements = [
        mpatches.Patch(color='#d7191c', label='Not suitable'),
        mpatches.Patch(color='#fdae61', label='Less suitable'),
        mpatches.Patch(color='#ffffbf', label='Moderately suitable'),
        mpatches.Patch(color='#a6d96a', label='Suitable'),
        mpatches.Patch(color='#1a9641', label='Highly suitable')
    ]
    ax.legend(handles=legend_elements, loc='lower center', fontsize=8, bbox_to_anchor=(0.5, -0.2), ncol=5)
    
    ax.set_title(f'{title}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Easting (meters)')
    ax.set_ylabel('Northing (meters)')
    
    # Create scale bar
    scalebar = ScaleBar(1000, location='lower center', box_alpha=0.5, scale_loc='bottom', length_fraction=0.25, units='km')
    ax.add_artist(scalebar)
    
    ax.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction', 
                horizontalalignment='center', verticalalignment='center',
                fontsize=16, fontweight='bold', path_effects=[pe.withStroke(linewidth=3, foreground="w")])
    ax.arrow(0.95, 0.93, 0, 0.02, head_width=0.01, head_length=0.01, 
             fc='k', ec='k', transform=ax.transAxes)
    
    fig.patch.set_linewidth(2)
    fig.patch.set_edgecolor('black')
    
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y/1000:.0f}'))
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)
    
    # Add coordinate system info outside the map
    map_info = 'Coordinate system: Arc 1960 UTM Zone 37S\nProjection: Transverse Mercator\nDatum: Arc 1960'
    plt.figtext(0.95, 0.01, map_info, fontsize=6, ha='right', va='bottom')
    
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
    
    cell_size = 50  # Define cell size in meters (reduced for higher resolution)
    grid_gdf, x_coords, y_coords = create_grid(aoi_boundary, cell_size=cell_size, socketio=socketio, session_id=session_id)
    
    if grid_gdf is None or grid_gdf.empty:
        emit_progress("Grid creation failed. Aborting optimization process.", socketio, session_id)
        return None
    
    grid_gdf = calculate_proximity(grid_gdf, roads_gdf, settlements_gdf, aoi_boundary, socketio, session_id)
    
    create_suitability_map(grid_gdf, 'road_score', "Road Suitability", "output/road_suitability.png", aoi_boundary, socketio, session_id)
    create_suitability_map(grid_gdf, 'settlement_score', "Settlement Suitability", "output/settlement_suitability.png", aoi_boundary, socketio, session_id)
    create_suitability_map(grid_gdf, 'combined_score', "Combined Suitability", "output/combined_suitability.png", aoi_boundary, socketio, session_id)
    
    grid_gdf = perform_clustering(grid_gdf, n_clusters=5, socketio=socketio, session_id=session_id)
    if grid_gdf is None:
        emit_progress("Clustering failed. Aborting optimization process.", socketio, session_id)
        return None
    
    optimal_locations = select_optimal_locations(grid_gdf, socketio, session_id)
    
    emit_progress("Creating final map with optimal locations", socketio, session_id)
    try:
        fig, ax = plt.subplots(figsize=(20, 20))
        
        aoi_boundary.plot(ax=ax, color='none', edgecolor='black', linewidth=1, zorder=1)
        
        grid_gdf.plot(column='combined_score', ax=ax, cmap='RdYlGn', 
                       
                      alpha=0.6, zorder=2)
        
        scatter = ax.scatter(
            grid_gdf.geometry.centroid.x, grid_gdf.geometry.centroid.y,
            c=grid_gdf['combined_score'], cmap='RdYlGn',
            s=5, alpha=0.6, zorder=2
        )
        
        roads_gdf.plot(ax=ax, color='gray', linewidth=0.5, zorder=3, label='Roads')
        settlements_gdf.plot(ax=ax, color='blue', markersize=10, alpha=0.7, zorder=4, label='Settlements')
        
        if not optimal_locations.empty:
            optimal_locations.plot(ax=ax, color='red', markersize=50, alpha=0.7, zorder=5, label="Optimal locations")
    
        # Add color bar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.1)  # Reduced size
        cbar = plt.colorbar(scatter, cax=cax, label='Suitability Score')
        cbar.set_ticks([1, 2, 3, 4, 5])
        cbar.set_ticklabels(['Not Suitable', 'Less Suitable', 'Moderately Suitable', 'Suitable', 'Highly Suitable'])

        ax.set_title('Optimal Waste Collection Points', fontsize=22, fontweight="bold")
        ax.set_xlabel('Easting (meters)')
        ax.set_ylabel('Northing (meters)')

        ax.grid(True, linestyle='--', alpha=0.7)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}'))

        ax.legend(loc='lower left', bbox_to_anchor=(0, 0), fontsize=12)

        scalebar = ScaleBar(1, location='lower right', box_alpha=0.5, scale_loc='bottom')
        ax.add_artist(scalebar)
        
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
        
        x, y, arrow_length = 0.95, 0.95, 0.05
        ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                    arrowprops=dict(facecolor='black', width=5, headwidth=15),
                    ha='center', va='center', fontsize=12,
                    xycoords=ax.transAxes,
                    path_effects=[pe.withStroke(linewidth=3, foreground="w")])

        map_info = (
            'Coordinate System: Arc 1960 UTM Zone 37S\n'
            'Projection: Transverse Mercator\n'
            'Datum: Arc 1960\n'
            'Units: Meters'
        )
        plt.text(0.02, 0.02, map_info, transform=ax.transAxes, fontsize=8, 
                 verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

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

    socketio.enter_room(session_id)
    
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

if __name__ == '__main__':
    socketio.run(app, debug=True)