import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, box
import rasterio
import os
import sys
import traceback
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patheffects as pe
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime
import eventlet
import joblib
from scipy.ndimage import gaussian_filter, zoom
from sklearn.cluster import KMeans, DBSCAN
from rasterio.plot import plotting_extent
from rasterio.mask import mask
import rasterio.transform
from flask import Flask
from flask_socketio import SocketIO
from sqlalchemy import create_engine
import logging
from pyproj import CRS
import xlsxwriter
from shapely.ops import unary_union
from shapely.validation import make_valid
import warnings
import datetime
from sqlalchemy import text
# Import contextily for basemap

import contextily as ctx
        

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
    
    # grid_gdf = validate_geometries(grid_gdf, socketio, session_id)
    
    grid_gdf['is_within_aoi'] = grid_gdf.within(aoi_gdf.unary_union)
    
    grid_gdf['x_idx'] = ((grid_gdf.geometry.bounds['minx'] - xmin) // cell_size).astype(int)
    grid_gdf['y_idx'] = ((grid_gdf.geometry.bounds['miny'] - ymin) // cell_size).astype(int)
    
    emit_progress(f"Total grid cells created: {len(grid_gdf)}", socketio, session_id)
    emit_progress(f"Grid cells within AOI: {grid_gdf['is_within_aoi'].sum()}", socketio, session_id)
    
    return grid_gdf, x_coords, y_coords
def calculate_proximity(grid_gdf, roads_gdf, settlements_gdf, aoi_gdf, socketio=None, session_id=None):
    emit_progress("Calculating proximity to roads and settlements", socketio, session_id)
    
    try:
        emit_progress("Clipping grid to AOI", socketio, session_id)
        grid_gdf = gpd.clip(grid_gdf, aoi_gdf)
        
        emit_progress("Validating geometries of roads and settlements.", socketio, session_id)
        roads_gdf = validate_geometries(roads_gdf, socketio, session_id)
        settlements_gdf = validate_geometries(settlements_gdf, socketio, session_id)
        
        emit_progress("Creating road suitability buffers", socketio, session_id)
        
        # Create a unified road geometry
        road_union = roads_gdf.geometry.unary_union
        if road_union is None or road_union.is_empty:
            emit_progress("Warning: No valid road geometries found", socketio, session_id)
            return grid_gdf
            
        # Create buffers with error handling
        try:
            buffer_0 = road_union.buffer(0)
            buffer_50 = road_union.buffer(50)
            buffer_200 = road_union.buffer(200)
            buffer_500 = road_union.buffer(500)
            buffer_1000 = road_union.buffer(1000)
            
            road_buffers = []
            
            # Zone 1: 0-50m
            if not buffer_50.is_empty:
                zone1 = buffer_50.difference(buffer_0)
                if not zone1.is_empty:
                    road_buffers.append((zone1, 5))
                    
            # Zone 2: 50-200m
            if not buffer_200.is_empty:
                zone2 = buffer_200.difference(buffer_50)
                if not zone2.is_empty:
                    road_buffers.append((zone2, 4))
                    
            # Zone 3: 200-500m
            if not buffer_500.is_empty:
                zone3 = buffer_500.difference(buffer_200)
                if not zone3.is_empty:
                    road_buffers.append((zone3, 3))
                    
            # Zone 4: 500-1000m
            if not buffer_1000.is_empty:
                zone4 = buffer_1000.difference(buffer_500)
                if not zone4.is_empty:
                    road_buffers.append((zone4, 2))
                    
            # Zone 5: Beyond 1000m
            zone5 = aoi_gdf.geometry.unary_union.difference(buffer_1000)
            if not zone5.is_empty:
                road_buffers.append((zone5, 1))
                
            # Create GeoDataFrame from valid buffers
            if road_buffers:
                road_suitability = gpd.GeoDataFrame(
                    {'geometry': [geom for geom, _ in road_buffers],
                     'road_score': [score for _, score in road_buffers]},
                    crs=aoi_gdf.crs
                )
            else:
                emit_progress("Warning: No valid buffer zones created", socketio, session_id)
                road_suitability = gpd.GeoDataFrame(
                    {'geometry': [aoi_gdf.geometry.unary_union],
                     'road_score': [1]},
                    crs=aoi_gdf.crs
                )
            
        except Exception as e:
            emit_progress(f"Error creating buffer zones: {str(e)}", socketio, session_id)
            logger.error(f"Buffer creation error: {str(e)}")
            return grid_gdf
            
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
        
    except Exception as e:
        emit_progress(f"Error in calculate_proximity: {str(e)}", socketio, session_id)
        logger.error(f"Error in calculate_proximity: {str(e)}")
        traceback.print_exc()
        return grid_gdf

#SAVE COLLECTION POINTS TO DB 
def save_collection_points_to_db(markers_data, engine, socketio=None, session_id=None):
    """
    Save or update collection points in the database
    """
    try:
        # Create points from coordinates
        collection_points = []
        for marker in markers_data:
            point_geom = f"SRID=21037;POINT({marker['x_coord']} {marker['y_coord']})"
            collection_points.append({
                'point_id': marker['site_id'],
                'description': f'Collection Point {marker["site_id"]}',
                'geom': point_geom,
                'updated_at': datetime.datetime.now()
            })

        # Create a temporary table for the new points
        temp_table_query = text("""
        CREATE TEMP TABLE temp_collection_points (
            point_id INT,
            description TEXT,
            geom geometry(Point, 21037),
            updated_at TIMESTAMP
        ) ON COMMIT DROP;
        """)
        
        # Insert new points into temp table
        insert_temp_query = text("""
        INSERT INTO temp_collection_points (point_id, description, geom, updated_at)
        VALUES (:point_id, :description, :geom::geometry, :updated_at);
        """)
        
        # Upsert query using the temp table
        upsert_query = text("""
        INSERT INTO collection_points (point_id, description, geom, created_at, updated_at)
        SELECT 
            t.point_id,
            t.description,
            t.geom,
            COALESCE(cp.created_at, CURRENT_TIMESTAMP),
            t.updated_at
        FROM temp_collection_points t
        LEFT JOIN collection_points cp ON t.point_id = cp.point_id
        ON CONFLICT (point_id) 
        DO UPDATE SET
            description = EXCLUDED.description,
            geom = EXCLUDED.geom,
            updated_at = EXCLUDED.updated_at
        RETURNING id, point_id;
        """)

        with engine.begin() as connection:
            # Create temporary table
            connection.execute(temp_table_query)
            
            # Insert into temp table
            for point in collection_points:
                connection.execute(insert_temp_query, point)
            
            # Perform upsert and get results
            result = connection.execute(upsert_query)
            affected_rows = result.rowcount

        emit_progress(f"Successfully saved/updated {affected_rows} collection points to database", 
                     socketio, session_id)
        return True

    except Exception as e:
        emit_progress(f"Error saving collection points to database: {str(e)}", 
                     socketio, session_id)
        logger.error(f"Database error: {str(e)}")
        return False

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
    
def create_markers_map(grid_gdf, aoi_gdf,  engine=None,socketio=None, session_id=None):
    """
    Create a simple map showing only collection points without basemap
    """
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(20, 20))
        
        # Filter highly suitable areas
        highly_suitable = grid_gdf[grid_gdf['Suitability_Class'] == 'Highly Suitable']
        
        # Perform DBSCAN clustering
        coords = np.array([[geom.centroid.x, geom.centroid.y] 
                          for geom in highly_suitable.geometry])
        
        clustering = DBSCAN(
            eps=50,  # 50 meters clustering distance
            min_samples=1
        ).fit(coords)
        
        highly_suitable['cluster'] = clustering.labels_
        
        # Plot AOI boundary
        aoi_gdf.boundary.plot(ax=ax, color='black', linewidth=1.5, alpha=0.7)
        
        # Create markers for each cluster
        markers_data = []
        for cluster_id in range(clustering.labels_.max() + 1):
            cluster_points = highly_suitable[highly_suitable['cluster'] == cluster_id]
            cluster_centroid = cluster_points.geometry.unary_union.centroid
            
            # Plot marker as a pin
            ax.plot(cluster_centroid.x, cluster_centroid.y, 
                   marker='^',  # Triangle marker for pin appearance
                   color='red',
                   markersize=15,
                   markeredgecolor='black',
                   markeredgewidth=1,
                   zorder=5)
            
            # Simple label with just the site number
            ax.annotate(f'Site {len(markers_data) + 1}',
                       (cluster_centroid.x, cluster_centroid.y),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=10,
                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
            
            markers_data.append({
                'site_id': len(markers_data) + 1,
                'x_coord': cluster_centroid.x,
                'y_coord': cluster_centroid.y
            })
        
        # Add simple legend
        ax.legend(['Area Boundary', 'Collection Points'], 
                 loc='upper right',
                 bbox_to_anchor=(1.1, 1))
        
        # Add north arrow
        ax.annotate('N', xy=(0.98, 0.98), xycoords='axes fraction',
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=18, fontweight='bold',
                   path_effects=[pe.withStroke(linewidth=3, foreground="w")])
        
        # Add scale bar
        scalebar = ScaleBar(1, location='lower right')
        ax.add_artist(scalebar)
        
        # Add gridlines
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Format axis labels
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y/1000:.0f}'))
        
        # Add title
        ax.set_title('Waste Collection Points', fontsize=16, pad=20)
        
        # Add coordinate system info
        map_info = (
            'Coordinate System: Arc 1960 UTM Zone 37S\n'
            'Projection: Transverse Mercator\n'
            'Datum: Arc 1960\n'
            'Units: Meters'
        )
        plt.text(0.02, 0.02, map_info, transform=ax.transAxes, fontsize=8,
                verticalalignment='bottom',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Save map
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        map_path = os.path.join('output', f'collection_points_simple_{timestamp}.png')
        plt.savefig(map_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create and save markers data to CSV and Excel
        markers_df = pd.DataFrame(markers_data)
        

        success = save_collection_points_to_db(markers_data, engine, socketio, session_id)
        
        if not success:
            emit_progress("Failed to save collection points to database", 
                         socketio, session_id)
            return None, None
        
        return map_path, pd.DataFrame(markers_data)
        
    except Exception as e:
        emit_progress(f"Error in create_markers_map: {str(e)}", 
                     socketio, session_id)
        logger.error(f"Error in create_markers_map: {str(e)}")
        return None, None
    
def create_suitability_map(grid_gdf, score_column, title, output_path, aoi_gdf, socketio=None, session_id=None):
    emit_progress(f"Creating suitability map: {title}", socketio, session_id)
    
    try:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create custom colormap
        cmap = colors.LinearSegmentedColormap.from_list("custom", 
            ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641'], N=256)
        
        # Create normalization
        norm = colors.BoundaryNorm([1, 2, 3, 4, 5, 6], cmap.N)
        
        # Plot suitability grid using GeoDataFrame's plot method
        grid_gdf.plot(
            column=score_column,
            ax=ax,
            cmap=cmap,
            legend=True,
            legend_kwds={
                'label': 'Suitability Score',
                'orientation': 'vertical'
            }
        )
        
        # Plot area of interest boundary
        aoi_gdf.boundary.plot(
            ax=ax,
            edgecolor='black',
            linewidth=1.5,
            label='Area of Interest Boundary'
        )
        
        # Set title and labels
        ax.set_title(f'{title}', fontsize=18, fontweight='bold', y=1.05)
        ax.set_xlabel('Easting (meters)', fontsize=12)
        ax.set_ylabel('Northing (meters)', fontsize=12)
        
        # Add north arrow
        ax.annotate('N', xy=(0.98, 0.98), xycoords='axes fraction', 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=18, fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=3, foreground="w")])
        ax.arrow(0.98, 0.96, 0, 0.02, head_width=0.01, head_length=0.01, 
                 fc='k', ec='k', transform=ax.transAxes)
        
        # Set gridlines and ticks
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y/1000:.0f}'))
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Add custom legend
        legend_elements = [
            mpatches.Patch(color='#d7191c', label='Not suitable'),
            mpatches.Patch(color='#fdae61', label='Less suitable'),
            mpatches.Patch(color='#ffffbf', label='Moderately suitable'),
            mpatches.Patch(color='#a6d96a', label='Suitable'),
            mpatches.Patch(color='#1a9641', label='Highly suitable')
        ]
        ax.legend(handles=legend_elements, loc='center left', 
                 bbox_to_anchor=(1.05, 0.5), title='Suitability Classes')
        
        # Add scale bar
        scalebar = ScaleBar(1, location='lower right', box_alpha=0.5)
        ax.add_artist(scalebar)
        
        # Add map information
        map_info = (
            'Coordinate System: Arc 1960 UTM Zone 37S\n'
            'Projection: Transverse Mercator\n'
            'Datum: Arc 1960\n'
            'Units: Meters'
        )
        plt.text(0.02, 0.02, map_info, transform=ax.transAxes, fontsize=8,
                verticalalignment='bottom',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        emit_progress(f"Suitability map saved: {output_path}", socketio, session_id)
        logger.info(f"Suitability map saved: {output_path}")
        
    except Exception as e:
        emit_progress(f"Error creating suitability map: {str(e)}", socketio, session_id)
        logger.error(f"Error creating suitability map: {str(e)}")
        traceback.print_exc()
        
        
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
    
    # Validate scores before creating maps
    if 'road_score' not in grid_gdf.columns:
        emit_progress("Error: road_score column missing from grid", socketio, session_id)
        return None
    
    if 'settlement_score' not in grid_gdf.columns:
        emit_progress("Error: settlement_score column missing from grid", socketio, session_id)
        return None
    
    if 'combined_score' not in grid_gdf.columns:
        emit_progress("Error: combined_score column missing from grid", socketio, session_id)
        return None
    
    create_suitability_map(grid_gdf, 'road_score', "Road Suitability", "output/road_suitability.png", aoi_boundary, socketio, session_id)
    create_suitability_map(grid_gdf, 'settlement_score', "Settlement Suitability", "output/settlement_suitability.png", aoi_boundary, socketio, session_id)
    create_suitability_map(grid_gdf, 'combined_score', "Combined Suitability", "output/combined_suitability.png", aoi_boundary, socketio, session_id)
    
    grid_gdf['Suitability_Class'] = pd.cut(
        grid_gdf['combined_score'],
        bins=[0, 2, 2.75, 3.5, 4.25, 5],
        labels=['Not Suitable', 'Less Suitable', 'Moderately Suitable', 'Suitable', 'Highly Suitable']
    )
    
       
    markers_map_path, markers_df = create_markers_map(
        grid_gdf, aoi_boundary, engine, socketio, session_id
    )
    if markers_df is not None and not markers_df.empty:
       emit_progress(f"Created markers map with {len(markers_df)} collection points", socketio, session_id)
    else:
       emit_progress("No suitable collection points found", socketio, session_id)
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
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y/1000:.0f}'))

        ax.legend(loc='lower left',  fontsize=12, bbox_to_anchor=(0, 0.5))

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
    'optimal_locations': 'output/optimal_locations_detailed.png',
    'markers_map': markers_map_path,
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