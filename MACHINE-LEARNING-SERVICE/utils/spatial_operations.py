import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from shapely.geometry import MultiPolygon, LineString
from shapely.ops import unary_union
from pyproj import CRS

def fetch_data_from_postgis(query, engine):
    return gpd.read_postgis(query, engine, geom_col='geom')

def create_arcgis_like_buffers(gdf, distances):
    if gdf.crs is None:
        print("Warning: GeoDataFrame has no CRS set. Assuming EPSG:4326.")
        gdf.set_crs(epsg=4326, inplace=True)
    
    utm_crs = CRS.from_epsg(32737)  # UTM zone 37S
    gdf_projected = gdf.to_crs(utm_crs)
    print(f"Projected CRS: {utm_crs.name}")

    buffered_gdfs = []
    for distance in distances:
        # Create a buffer for each feature with very smooth corners
        buffered_geoms = []
        for geom in gdf_projected.geometry:
            if geom.geom_type == 'MultiLineString':
                buffered = MultiPolygon([LineString(part).buffer(distance, cap_style=3, join_style=2, resolution=32) 
                                         for part in geom.geoms])
            else:
                buffered = geom.buffer(distance, cap_style=3, join_style=2, resolution=32)
            buffered_geoms.append(buffered)
        
        # Merge all buffers into a single MultiPolygon
        merged_buffer = unary_union(buffered_geoms)
        
        # Create a new GeoDataFrame with the merged buffer
        buffered_gdf = gpd.GeoDataFrame(geometry=[merged_buffer], crs=utm_crs)
        buffered_gdf['buffer_distance'] = distance
        
        print(f"Created buffer of {distance} meters")
        buffered_gdf = buffered_gdf.to_crs(gdf.crs)
        buffered_gdfs.append(buffered_gdf)
    return buffered_gdfs
def plot_buffers(buffered_gdfs, original_gdf, ax=None, title='Buffers', color='blue', feature_color='darkblue'):
    if ax is None:
        _, ax = plt.subplots(figsize=(15, 15))
    
    legend_patches = []
    for i, buffered_gdf in enumerate(reversed(buffered_gdfs)):
        if not buffered_gdf.empty:
            alpha = 0.1 + (0.2 * i)  # Vary opacity from 0.1 to 0.7
            buffered_gdf.plot(ax=ax, edgecolor='none', facecolor=color, alpha=alpha, aspect=None)
            legend_patches.append(mpatches.Patch(facecolor=color, alpha=alpha, 
                                  label=f'{buffered_gdf["buffer_distance"].iloc[0]}m Buffer'))
    
    original_gdf.plot(ax=ax, color=feature_color, linewidth=1, zorder=5, aspect=None)
    legend_patches.append(mlines.Line2D([], [], color=feature_color, linewidth=1, label='Original Geometry'))
    
    ax.set_title(title, fontsize=16)
    ax.legend(handles=legend_patches, title='Buffer Distances', 
              loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    
    ax.set_axis_off()
    
    return ax