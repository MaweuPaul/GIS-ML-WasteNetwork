from utils.spatial_operations import fetch_data_from_postgis, create_arcgis_like_buffers, plot_buffers
from sqlalchemy import create_engine, text

# Connection details to talk to the database
connection_params = {
    'dbname': 'GEGIS',
    'user': 'postgres',
    'password': 'p#maki012412',
    'host': 'localhost',
    'port': '5432'
}

# Create SQLAlchemy engine
engine = create_engine(f"postgresql://{connection_params['user']}:{connection_params['password']}@{connection_params['host']}:{connection_params['port']}/{connection_params['dbname']}")

# Function to get the correct table name
def get_table_name(engine):
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name ILIKE 'river'
        """))
        row = result.fetchone()
        return row[0] if row else None

# Get the correct table name
table_name = get_table_name(engine)

if table_name:
    # SQL to get the river shapes
    rivers_query = f'SELECT id, geom FROM "{table_name}"'

    try:
        # Get the river shapes
        rivers_gdf = fetch_data_from_postgis(rivers_query, engine)
        print(f"Fetched {len(rivers_gdf)} river geometries")

        # Diagnostic information
        print(f"Original CRS: {rivers_gdf.crs}")
        print(f"Geometry types: {rivers_gdf.geometry.type.unique()}")
        print(f"Number of features: {len(rivers_gdf)}")

        # Make circles (buffers) around the rivers
        buffer_distances = [200, 400, 800, 1200]  # Distances in meters
        rivers_buffers = create_arcgis_like_buffers(rivers_gdf, buffer_distances)

        # Show the circles on a map
        plot_buffers(rivers_buffers, rivers_gdf)
    except Exception as e:
        print(f"Error: {e}")
else:
    print("River table not found in the database.")