
from utils.spatial_operations import fetch_data_from_postgis, create_arcgis_like_buffers, plot_buffers
from sqlalchemy import create_engine, text
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from rasterio.plot import plotting_extent 
from shapely import wkt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Connection details to talk to the database
connection_params = {
    'dbname': 'GEGIS',
    'user': 'postgres',
    'password': 'p#maki012412',
    'host': 'localhost',
    'port': '5432'
}

# Create SQLAlchemy engine
engine = create_engine(
    f"postgresql://{connection_params['user']}:{connection_params['password']}@"
    f"{connection_params['host']}:{connection_params['port']}/{connection_params['dbname']}"
)

def get_table_name(engine, table_type):
    with engine.connect() as conn:
        result = conn.execute(text(f"""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name ILIKE '{table_type}'
        """))
        row = result.fetchone()
        return row[0] if row else None

# Function to fetch and buffer data
def fetch_and_buffer(engine, table_name, buffer_distances):
    query = f'SELECT id, geom FROM "{table_name}"'
    try:
        gdf = fetch_data_from_postgis(query, engine)
        print(f"Fetched {len(gdf)} geometries from {table_name}")
        print(f"Original CRS: {gdf.crs}")
        print(f"Geometry types: {gdf.geometry.type.unique()}")
        print(f"Number of features: {len(gdf)}")
        
        buffers = create_arcgis_like_buffers(gdf, buffer_distances)
        return gdf, buffers
    except Exception as e:
        print(f"Error processing {table_name}: {e}")
        return None, None

# Fetch and buffer data for rivers, roads, protected areas, and settlements
river_distances = [200, 500, 1000, 1500]
road_distances = [200, 500, 1000]
protected_area_distances = [200, 500, 1000, 1500]
settlement_distances = [200, 500, 1000, 1500]

rivers_gdf, rivers_buffers = fetch_and_buffer(engine, 'River', river_distances)
roads_gdf, roads_buffers = fetch_and_buffer(engine, 'Road', road_distances)
protected_areas_gdf, protected_areas_buffers = fetch_and_buffer(engine, 'ProtectedArea', protected_area_distances)
settlements_gdf, settlements_buffers = fetch_and_buffer(engine, 'Settlement', settlement_distances)

# Visualize the buffer results
fig, ax = plt.subplots(figsize=(15, 15))

if all([rivers_buffers, roads_buffers, protected_areas_buffers, settlements_buffers]):
    plot_buffers(rivers_buffers, rivers_gdf, ax=ax, title='Combined Buffers', color='lightblue', feature_color='blue')
    plot_buffers(roads_buffers, roads_gdf, ax=ax, color='lightgreen', feature_color='green')
    plot_buffers(protected_areas_buffers, protected_areas_gdf, ax=ax, color='pink', feature_color='red')
    plot_buffers(settlements_buffers, settlements_gdf, ax=ax, color='plum', feature_color='purple')
    ax.set_title('Combined Buffer Zones', fontsize=16)
    ax.set_aspect('equal')

plt.tight_layout()
plt.show()

# Now, let's move on to the DEM and slope analysis

# Function to fetch DEM data and convert to GeoDataFrame
def fetch_dem_data(engine):
    query = """
    SELECT id, name, elevation, "geometryType", 
           ST_AsText(geom) as geom_wkt
    FROM "DigitalElevationModel"
    """
    try:
        with engine.connect() as connection:
            dem_df = pd.read_sql(query, connection)
        print(f"Fetched {len(dem_df)} DEM polygons")
        print(f"Geometry types: {dem_df['geometryType'].unique()}")
        print(f"Sample geometry WKT: {dem_df['geom_wkt'].iloc[0][:100]}...")  
        
        # Convert to GeoDataFrame
        dem_gdf = gpd.GeoDataFrame(
            dem_df,
            geometry=gpd.GeoSeries.from_wkt(dem_df['geom_wkt']),
            crs=rivers_gdf.crs  # Assuming DEM has the same CRS as rivers_gdf
        )
        return dem_gdf
    except Exception as e:
        print(f"Error fetching DEM data: {e}")
        return None

# Fetch DEM data
dem_gdf = fetch_dem_data(engine)

# Check if DEM data was successfully fetched
if dem_gdf is None or dem_gdf.empty:
    print("Failed to fetch DEM data. Skipping DEM and slope analysis.")
else:
    # Reproject DEM GeoDataFrame to a Projected CRS (e.g., UTM)
    try:
        # Choose an appropriate UTM zone based on the centroid of DEM data
        centroid = dem_gdf.unary_union.centroid
        utm_zone = int((centroid.x + 180) / 6) + 1
        # Determine if the location is in northern or southern hemisphere
        if centroid.y >= 0:
            epsg_code = 32600 + utm_zone  # Northern Hemisphere
        else:
            epsg_code = 32700 + utm_zone  # Southern Hemisphere
        projected_crs = f"EPSG:{epsg_code}"
        print(f"Reprojecting DEM data to {projected_crs}")
        
        dem_gdf_projected = dem_gdf.to_crs(projected_crs)
        print(f"DEM data reprojected to {projected_crs}")
    except Exception as e:
        print(f"Error reprojecting DEM data: {e}")
        dem_gdf_projected = dem_gdf  # Fallback to original CRS

    # Reproject buffer layers to match DEM projected CRS
    try:
        if rivers_gdf.crs != dem_gdf_projected.crs:
            rivers_gdf = rivers_gdf.to_crs(dem_gdf_projected.crs)
            print("Rivers reprojected to match DEM CRS.")
        if roads_gdf.crs != dem_gdf_projected.crs:
            roads_gdf = roads_gdf.to_crs(dem_gdf_projected.crs)
            print("Roads reprojected to match DEM CRS.")
        if protected_areas_gdf.crs != dem_gdf_projected.crs:
            protected_areas_gdf = protected_areas_gdf.to_crs(dem_gdf_projected.crs)
            print("Protected Areas reprojected to match DEM CRS.")
        if settlements_gdf.crs != dem_gdf_projected.crs:
            settlements_gdf = settlements_gdf.to_crs(dem_gdf_projected.crs)
            print("Settlements reprojected to match DEM CRS.")
    except Exception as e:
        print(f"Error reprojecting buffer layers: {e}")

    # Function to convert DEM polygons to a raster grid using rasterize
    def create_dem_raster_rasterize(dem_gdf, resolution=30):
        # Determine the bounds of the DEM data
        minx, miny, maxx, maxy = dem_gdf.total_bounds
        print(f"DEM Bounds: minx={minx}, miny={miny}, maxx={maxx}, maxy={maxy}")
        
        # Calculate the number of cells
        width = int((maxx - minx) / resolution)
        height = int((maxy - miny) / resolution)
        
        print(f"Raster Dimensions: width={width}, height={height}")
        
        if width < 2 or height < 2:
            raise ValueError(f"Raster size too small: width={width}, height={height}")
        
        # Create the transform
        transform = from_bounds(minx, miny, maxx, maxy, width, height)
        
        # Prepare shapes with elevation values
        shapes = ((geom, value) for geom, value in zip(dem_gdf.geometry, dem_gdf['elevation']))
        
        # Rasterize the DEM
        dem_raster = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=np.nan,  # Use NaN for areas without data
            dtype='float32',
            all_touched=True  # Include all pixels touched by geometries
        )
        
        return dem_raster, transform

    try:
        # Create DEM raster using rasterize method
        dem_raster, dem_transform = create_dem_raster_rasterize(dem_gdf_projected, resolution=30)
        print("DEM rasterization successful.")
        
        # Check the DEM raster
        print(f"DEM Raster Shape: {dem_raster.shape}")
        print(f"DEM Raster Stats: min={np.nanmin(dem_raster)}, max={np.nanmax(dem_raster)}, mean={np.nanmean(dem_raster)}")
        
        # Handle NaN values by filling them (e.g., with nearest valid value or a constant)
        from scipy.ndimage import generic_filter

        def fill_nan(array):
            # Replace NaN with the mean of the neighborhood
            if np.isnan(array).all():
                return np.nan
            return np.nanmean(array)
        
        dem_raster_filled = generic_filter(dem_raster, fill_nan, size=3, mode='nearest')
        print("NaN values in DEM raster have been filled.")
        
        # Calculate slope
        def calculate_slope(dem, transform):
            dx, dy = np.gradient(dem)
            cellsize_x = transform.a  # transform[0]
            cellsize_y = -transform.e  # transform[4]
            
            slope_rad = np.arctan(np.sqrt((dx / cellsize_x)**2 + (dy / cellsize_y)**2))
            slope_deg = np.degrees(slope_rad)
            
            return slope_deg

        # Ensure DEM raster has sufficient size for gradient calculation
        if dem_raster_filled.shape[0] < 2 or dem_raster_filled.shape[1] < 2:
            raise ValueError("DEM raster is too small for gradient calculation.")
        
        slope = calculate_slope(dem_raster_filled, dem_transform)
        print("Slope calculation successful.")
        
        # Visualize the slope
        plt.figure(figsize=(15, 15))
        extent = plotting_extent(dem_raster_filled, dem_transform)
        plt.imshow(slope, cmap='terrain', vmin=0, vmax=45, extent=extent)
        plt.title('Slope Map', fontsize=16)
        plt.colorbar(label='Slope (degrees)')
        plt.axis('off')
        plt.show()
        
        # Save the slope raster
        with rasterio.open(
            'slope_map.tif',
            'w',
            driver='GTiff',
            height=slope.shape[0],
            width=slope.shape[1],
            count=1,
            dtype='float32',
            crs=dem_gdf_projected.crs,  # Ensure CRS matches the input data
            transform=dem_transform
        ) as dst:
            dst.write(slope, 1)
        
        print("Slope analysis completed and results saved.")
    
    except ValueError as ve:
        print(f"ValueError during DEM rasterization or slope calculation: {ve}")
    except AttributeError as ae:
        print(f"AttributeError: {ae}. Please ensure that rasterio.plot is correctly imported.")
    except Exception as ex:
        print(f"An error occurred during DEM rasterization or slope calculation: {ex}")

print("All analyses completed.")