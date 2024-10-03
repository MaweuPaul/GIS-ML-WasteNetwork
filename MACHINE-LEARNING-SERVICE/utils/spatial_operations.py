import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from shapely.geometry import MultiPolygon, LineString
from pyproj import CRS
from shapely.ops import unary_union
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

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
                buffered = MultiPolygon([LineString(part).buffer(distance, cap_style=3, join_style=2, resolution=16) 
                                         for part in geom.geoms])
            else:
                buffered = geom.buffer(distance, cap_style=3, join_style=2, resolution=16)
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

def plot_buffers(buffered_gdfs, original_gdf, title='Rivers with Buffers'):
    print("Executing updated plot_buffers function")
    fig, ax = plt.subplots(figsize=(15, 15))
    colors = ['#FFA07A', '#98FB98', '#87CEFA', '#DDA0DD']
    
    legend_patches = []
    for i, buffered_gdf in enumerate(reversed(buffered_gdfs)):
        if not buffered_gdf.empty:
            color = colors[i]
            buffered_gdf.plot(ax=ax, alpha=0.5, edgecolor='none', facecolor=color)
            legend_patches.append(mpatches.Patch(color=color, alpha=0.5, 
                                  label=f'{buffered_gdf["buffer_distance"].iloc[0]}m Buffer'))
    
    original_gdf.plot(ax=ax, color='blue', linewidth=1, zorder=5)
    legend_patches.append(mlines.Line2D([], [], color='blue', linewidth=1, label='Original Geometry'))
    
    plt.title(title, fontsize=16)
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    
    # Create legend
    ax.legend(handles=legend_patches, title='Buffer Distances', 
              loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    
    ax.set_aspect('equal')
    plt.tight_layout()
    
    ax.annotate('N', xy=(0.02, 0.98), xytext=(0.02, 0.93),
                arrowprops=dict(facecolor='black', width=1, headwidth=8),
                ha='center', va='center', fontsize=20,
                xycoords=ax.transAxes)
    
    scalebar = AnchoredSizeBar(ax.transData,
                               0.01, '1 km', 'lower right', 
                               pad=0.5,
                               color='black',
                               frameon=False,
                               size_vertical=0.005)
    ax.add_artist(scalebar)
    
    plt.show()