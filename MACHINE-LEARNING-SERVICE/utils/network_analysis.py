import os
import datetime
import traceback
import eventlet
from shapely.geometry import Point, LineString, Polygon
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import folium
import osmnx as ox
from dotenv import load_dotenv
from sqlalchemy import create_engine
import json 
from shapely import wkt
import requests 

# Load environment variables
load_dotenv()

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

def emit_success(session_id, message, socketio):
    """Emit success messages with emoji"""
    try:
        formatted_message = f"✅ {message}"
        if socketio:
            socketio.emit('success_update', 
                         {'session_id': session_id, 'message': formatted_message}, 
                         room=session_id)
            eventlet.sleep(0)
        print(f"Success: {formatted_message}")
    except Exception as e:
        print(f"Error emitting success: {e}")
def get_landfill_sites_from_db(engine, session_id=None, socketio=None):
    """Fetch landfill sites from database"""
    try:
        emit_progress(session_id, "🔄 Fetching landfill sites from database...", socketio)
        
        query = """
        SELECT 
          id,
            landfill_id,
            suitability_score,
            suitability_class,
            ST_AsText(geom) as geometry
        FROM landfill_sites;
        """
        
        # Read from database
        df = pd.read_sql_query(query, engine)
        
        if df.empty:
            emit_error(session_id, "No landfill sites found in database", socketio)
            return None
            
        # Convert WKT to geometry
        df['geometry'] = df['geometry'].apply(lambda x: wkt.loads(x))
        
        # Add name column
        df['name'] = df.apply(lambda x: f'Landfill Site {x.landfill_id} ({x.suitability_class})', axis=1)
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:21037')
        
        emit_success(session_id, f"Successfully fetched {len(gdf)} landfill sites!", socketio)
        return gdf
        
    except Exception as e:
        emit_error(session_id, f"Error fetching landfill sites: {str(e)}", socketio)
        
def get_collection_points_from_db(engine, session_id=None, socketio=None):
    """Fetch collection points from database"""
    try:
        emit_progress(session_id, "🔄 Fetching collection points from database...", socketio)
        
        query = """
        SELECT 
            point_id,
            description,
            ST_AsText(geom) as geometry
        FROM collection_points;
        """
        
        # Read from database
        df = pd.read_sql_query(query, engine)
        
        if df.empty:
            emit_error(session_id, "No collection points found in database", socketio)
            return None
            
        # Convert WKT to geometry
        df['geometry'] = df['geometry'].apply(lambda x: wkt.loads(x))
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:21037')
        
        emit_success(session_id, f"Successfully fetched {len(gdf)} collection points!", socketio)
        return gdf
        
    except Exception as e:
        emit_error(session_id, f"Error fetching collection points: {str(e)}", socketio)
        return None

def create_dummy_data(session_id=None, socketio=None):
    """Create dummy data with 60 collection points and 2 landfill sites"""
    try:
        emit_progress(session_id, "🔄 Creating dummy data...", socketio)
        
        # Create a dummy study area (Nyeri boundary approximation)
        nyeri_coords = [
            (36.9, -0.45),
            (37.1, -0.45),
            (37.1, -0.35),
            (36.9, -0.35),
            (36.9, -0.45)
        ]
        nyeri_polygon = Polygon(nyeri_coords)
        nyeri_gdf = gpd.GeoDataFrame(
            {'name': ['Nyeri County']}, 
            geometry=[nyeri_polygon], 
            crs='EPSG:4326'
        )
        
        # Create collection points
        collection_points = []
        centers = [
            (37.0, -0.4),
            (37.05, -0.38),
            (36.95, -0.42),
            (37.08, -0.41)
        ]
        
        for i in range(60):
            center = centers[i % len(centers)]
            x = center[0] + np.random.normal(0, 0.01)
            y = center[1] + np.random.normal(0, 0.01)
            
            while not Point(x, y).within(nyeri_polygon):
                x = center[0] + np.random.normal(0, 0.01)
                y = center[1] + np.random.normal(0, 0.01)
            
            collection_points.append({
                'point_id': i + 1,
                'description': f'Collection Point {i + 1}',
                'geometry': Point(x, y)
            })
        
        collection_points_gdf = gpd.GeoDataFrame(
            collection_points,
            crs='EPSG:4326'
        )
        
        # Create landfill sites
        landfill_sites = [
            {
                'site_id': 1,
                'name': 'Main Landfill',
                'capacity': 5000,
                'geometry': Point(37.02, -0.39)
            },
            {
                'site_id': 2,
                'name': 'Secondary Landfill',
                'capacity': 3000,
                'geometry': Point(36.98, -0.41)
            }
        ]
        
        landfill_sites_gdf = gpd.GeoDataFrame(
            landfill_sites,
            crs='EPSG:4326'
        )
        
        emit_success(session_id, "🎉 Dummy data created successfully!", socketio)
        return nyeri_gdf, collection_points_gdf, landfill_sites_gdf
        
    except Exception as e:
        emit_error(session_id, f"Error creating dummy data: {str(e)} 😞", socketio)
        return None, None, None

def get_road_network_from_db(engine, bounds, session_id=None, socketio=None):
    """Fetch road network from database and convert to NetworkX graph"""
    try:
        emit_progress(session_id, "🔍 Starting road network fetch from database...", socketio)
        
        roads_query = f"""
        SELECT geom 
        FROM "Road"
        WHERE ST_Intersects(
            geom,
            ST_MakeEnvelope({bounds[0]}, {bounds[1]}, {bounds[2]}, {bounds[3]}, 4326)
        )
        """
        
        roads_gdf = gpd.read_postgis(
            roads_query,
            engine,
            geom_col='geom'
        )
        
        if roads_gdf.empty:
            emit_error(session_id, "🚫 No road data found in database.", socketio)
            return None
            
        if roads_gdf.crs != 'EPSG:21037':
            roads_gdf = roads_gdf.to_crs('EPSG:21037')
        
        G = nx.MultiDiGraph()
        
        total_segments = len(roads_gdf)
        for idx, row in roads_gdf.iterrows():
            if idx % 100 == 0:
                emit_progress(session_id, f"🛣️ Processing segment {idx + 1}/{total_segments}", socketio)
            
            coords = list(row.geometry.coords)
            
            for i in range(len(coords) - 1):
                start = coords[i]
                end = coords[i + 1]
                
                start_node = f"{start[0]:.6f},{start[1]:.6f}"
                end_node = f"{end[0]:.6f},{end[1]:.6f}"
                
                G.add_node(start_node, x=start[0], y=start[1])
                G.add_node(end_node, x=end[0], y=end[1])
                
                length = ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5
                
                G.add_edge(start_node, end_node, length=length)
                G.add_edge(end_node, start_node, length=length)
        
        emit_success(session_id, f"🗺️ Network created with {len(G.nodes)} nodes and {len(G.edges)} edges!", socketio)
        return G
        
    except Exception as e:
        emit_error(session_id, f"Error creating road network: {str(e)} 😞", socketio)
        traceback.print_exc()
        return None
    
def calculate_collection_point_distances(routes_gdf, session_id=None, socketio=None):
    """
    Calculate and format distances from landfill sites to collection points
    """
    try:
        # Data validation
        if routes_gdf.empty:
            emit_error(session_id, "Warning: Empty routes DataFrame", socketio)
            return pd.DataFrame(), {}

        required_columns = ['collection_point_id', 'landfill_id', 'distance_meters']
        if not all(col in routes_gdf.columns for col in required_columns):
            emit_error(session_id, "Warning: Missing required columns", socketio)
            return pd.DataFrame(), {}

        # Group routes by landfill and get routes to each collection point
        landfill_routes = []
        total_landfills = len(routes_gdf['landfill_id'].unique())
        
        for landfill_id in routes_gdf['landfill_id'].unique():
            landfill_data = routes_gdf[routes_gdf['landfill_id'] == landfill_id]
            
            # Progress update
            if socketio:
                emit_progress(
                    session_id, 
                    f"Processing landfill {landfill_id}/{total_landfills}", 
                    socketio
                )
            
            for _, route in landfill_data.iterrows():
                # Calculate time (assuming average speed of 40 km/h)
                distance_km = route['distance_meters'] / 1000
                time_minutes = distance_km * (60 / 40)
                
                landfill_routes.append({
                    'FROM': str(landfill_id),
                    'TO': str(route['collection_point_id']),
                    'DISTANCE(Km)': round(distance_km, 3),
                    'TIME(Minutes)': round(time_minutes, 2)
                })
        
        # Convert to DataFrame and sort by FROM and TO fields
        routes_df = pd.DataFrame(landfill_routes)
        if not routes_df.empty:
            routes_df = routes_df.sort_values(['FROM', 'TO'])
            
            # Calculate summary statistics
            summary_stats = {
                'average_distance_km': round(routes_df['DISTANCE(Km)'].mean(), 2),
                'total_distance_km': round(routes_df['DISTANCE(Km)'].sum(), 2),
                'total_time_minutes': round(routes_df['TIME(Minutes)'].sum(), 2),
                'number_of_routes': len(routes_df),
                'min_distance_km': round(routes_df['DISTANCE(Km)'].min(), 2),
                'max_distance_km': round(routes_df['DISTANCE(Km)'].max(), 2)
            }
            
            # Print summary statistics
            print("\nSummary Statistics:")
            for key, value in summary_stats.items():
                print(f"{key}: {value}")
            
            return routes_df, summary_stats
        else:
            emit_error(session_id, "No valid routes were calculated", socketio)
            return pd.DataFrame(), {}
            
    except Exception as e:
        emit_error(session_id, f"Error calculating distances: {str(e)}", socketio)
        return pd.DataFrame(), {}
def perform_network_analysis(nyeri_gdf, session_id=None, socketio=None, collection_points_gdf=None, engine=None):
    """
    Perform network analysis to find optimal routes between collection points and landfill sites
    """
    try:
        emit_progress(session_id, "🚀 Starting network analysis...", socketio)
        
        # Create output directory
        output_dir = os.path.join(os.getcwd(), 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Try to get collection points from database first
        collection_points_gdf = None
        if engine:
            try:
                collection_points_gdf = get_collection_points_from_db(engine, session_id, socketio)
                emit_success(session_id, "Successfully loaded collection points from database!", socketio)
            except Exception as e:
                emit_error(session_id, f"Failed to fetch from database: {str(e)}", socketio)
        
        # Get landfill sites from database
        landfill_sites_gdf = None
        if engine:
            landfill_sites_gdf = get_landfill_sites_from_db(engine, session_id, socketio)
        
        # Fall back to dummy data if needed
        if collection_points_gdf is None or landfill_sites_gdf is None:
            emit_progress(session_id, "Falling back to dummy data...", socketio)
            nyeri_gdf, collection_points_gdf, landfill_sites_gdf = create_dummy_data(session_id, socketio)
             
        # Ensure proper CRS and standardize
        for gdf, name in [(nyeri_gdf, 'nyeri'), 
                         (collection_points_gdf, 'collection points'),
                         (landfill_sites_gdf, 'landfill sites')]:
            if not gdf.crs:
                gdf.set_crs(epsg=4326, inplace=True)
                emit_progress(session_id, f"Set CRS for {name} to EPSG:4326", socketio)
            gdf.to_crs(epsg=4326, inplace=True)

        bounds = nyeri_gdf.total_bounds
        
        # Get road network with fallback
        try:
            emit_progress(session_id, "📥 Downloading road network from OpenStreetMap...", socketio)
            ox.config(timeout=300)
            G = ox.graph_from_bbox(
                north=bounds[3], 
                south=bounds[1],
                east=bounds[2], 
                west=bounds[0],
                network_type='drive',
                simplify=True
            )
            emit_success(session_id, "Successfully downloaded OSM network!", socketio)
        except Exception as e:
            emit_error(session_id, f"OSM download failed: {str(e)} 😞", socketio)
            
            if engine:
                G = get_road_network_from_db(engine, bounds, session_id, socketio)
                if not G:
                    raise Exception("Failed to get road network from both OSM and database. ❌")
            else:
                raise Exception("No database connection available for fallback. ❌")

        # Project everything to EPSG:21037
        G = ox.project_graph(G, to_crs='EPSG:21037')
        nyeri_gdf = nyeri_gdf.to_crs(epsg=21037)
        collection_points_gdf = collection_points_gdf.to_crs(epsg=21037)
        landfill_sites_gdf = landfill_sites_gdf.to_crs(epsg=21037)

        # Get nodes and edges
        nodes, edges = ox.graph_to_gdfs(G)
        
        # Find nearest nodes
        emit_progress(session_id, "🎯 Finding nearest network nodes...", socketio)
        collection_nodes = []
        landfill_nodes = []
        
        for idx, point in collection_points_gdf.iterrows():
            nearest = ox.distance.nearest_nodes(G, point.geometry.x, point.geometry.y)
            collection_nodes.append(nearest)
            
        for idx, site in landfill_sites_gdf.iterrows():
            centroid = site.geometry.centroid
            nearest = ox.distance.nearest_nodes(G, centroid.x, centroid.y)
            landfill_nodes.append(nearest)

        # Calculate routes
        emit_progress(session_id, "🛣️ Calculating optimal routes...", socketio)
        routes_data = []
        total_routes = len(landfill_nodes) * len(collection_nodes)
        completed_routes = 0
        
        for lf_idx, lf_node in enumerate(landfill_nodes):
            landfill_id = landfill_sites_gdf.iloc[lf_idx]['landfill_id']
            for cp_idx, cp_node in enumerate(collection_nodes):
                try:
                    route = nx.shortest_path(G, lf_node, cp_node, weight='length')
                    if len(route) < 2:
                        continue
                        
                    length = nx.shortest_path_length(G, lf_node, cp_node, weight='length')
                    
                    # Get coordinates for the route
                    route_coords = []
                    for node in route:
                        x = G.nodes[node]['x']
                        y = G.nodes[node]['y']
                        if (x, y) not in route_coords:
                            route_coords.append((x, y))
                    
                    if len(route_coords) >= 2:
                        routes_data.append({
                            'landfill_id': landfill_id,
                            'collection_point_id': collection_points_gdf.iloc[cp_idx]['point_id'],
                            'distance_meters': length,
                            'geometry': LineString(route_coords)
                        })
                    
                    completed_routes += 1
                    if completed_routes % 10 == 0:
                        progress = (completed_routes / total_routes) * 100
                        emit_progress(session_id, f"📈 Route calculation progress: {progress:.1f}% 🔄", socketio)
                        
                except nx.NetworkXNoPath:
                    emit_error(session_id, f"No route found between LF {landfill_id} and CP {collection_points_gdf.iloc[cp_idx]['point_id']} 🚫", socketio)
                except Exception as e:
                    emit_error(session_id, f"Error calculating route LF {landfill_id} to CP {collection_points_gdf.iloc[cp_idx]['point_id']}: {str(e)} 🚫", socketio)
                    continue
        
        # Create routes GeoDataFrame and calculate summaries
        if not routes_data:
            emit_error(session_id, "No valid routes were created", socketio)
            return None
            
        routes_gdf = gpd.GeoDataFrame(routes_data, crs='EPSG:21037')
        emit_progress(session_id, "📊 Calculating route summaries...", socketio)
        routes_summary, summary_stats = calculate_collection_point_distances(routes_gdf, session_id, socketio)
        
        if routes_summary.empty:
            emit_error(session_id, "Failed to calculate route summary", socketio)
            return None
        routes_for_db = []
 
        for idx, route in routes_gdf.iterrows():
            try:
                # Get the landfill details using the actual database ID
                landfill = landfill_sites_gdf[landfill_sites_gdf['landfill_id'] == route['landfill_id']]
                
                if landfill.empty:
                    emit_error(session_id, f"❌ Landfill not found for ID {route['landfill_id']}", socketio)
                    continue
                    
                
                actual_id = int(landfill.iloc[0]['id'])  
                
                emit_progress(session_id, f"Mapping landfill_id {route['landfill_id']} to actual ID {actual_id}", socketio)
        
                route_data = {
                    'collection_point_id': int(route['collection_point_id']),
                    'landfill_id': actual_id,  
                    'distance_meters': float(route['distance_meters']),
                    'geometry': {
                        'type': 'LineString',
                        'coordinates': [[float(x), float(y)] for x, y in route.geometry.coords],
                        'crs': {
                            'type': 'name',
                            'properties': {
                                'name': 'EPSG:21037'
                            }
                        }
                    }
                }
        
                routes_for_db.append(route_data)
        
            except Exception as e:
                emit_error(session_id, f"❌ Error processing route {idx}: {str(e)}", socketio)
                continue
        
        # Verify the first few routes
        emit_progress(session_id, "Routes verification:", socketio)
        for i in range(min(3, len(routes_for_db))):
            emit_progress(
                session_id, 
                f"Route {i + 1}: collection_point={routes_for_db[i]['collection_point_id']}, " +
                f"landfill={routes_for_db[i]['landfill_id']} (should be 22, 23, or 24)", 
                socketio
            )
        # Validate final payload
        if not routes_for_db:
            emit_error(session_id, "❌ No valid routes to save", socketio)
            return
        
        payload = {
            'routes': routes_for_db
        }
        
    
        emit_progress(session_id, f"Total routes to save: {len(routes_for_db)}", socketio)
        
        # Send to API with detailed error handling
        try:
            emit_progress(session_id, "💾 Saving routes to database...", socketio)
            
            response = requests.post(
                'http://localhost:3000/api/waste-management/save-routes',
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            
            if response.status_code == 200:
                emit_success(session_id, f"✅ Successfully saved {len(routes_for_db)} routes!", socketio)
            else:
                try:
                    error_data = response.json()
                    error_message = error_data.get('error', 'Unknown error')
                    emit_error(session_id, f"Failed to save routes: {error_message}", socketio)
                except ValueError:
                    emit_error(session_id, f"Failed to save routes: {response.text}", socketio)
                
        except requests.exceptions.ConnectionError:
            emit_error(session_id, "❌ Could not connect to the API server", socketio)
        except requests.exceptions.Timeout:
            emit_error(session_id, "❌ Request timed out", socketio)
        except requests.exceptions.RequestException as e:
            emit_error(session_id, f"❌ Request error: {str(e)}", socketio)
        except Exception as e:
            emit_error(session_id, f"❌ Unexpected error: {str(e)}", socketio)
        # Save outputs
        file_paths = {}
        
        # Save routes summary
        summary_filename = f'routes_summary_session_{session_id}_{timestamp}.csv'
        summary_path = os.path.join(output_dir, summary_filename)
        routes_summary.to_csv(summary_path, index=False)
        file_paths['routes_summary'] = summary_path
        
        # Save summary statistics
        stats_filename = f'summary_stats_session_{session_id}_{timestamp}.json'
        stats_path = os.path.join(output_dir, stats_filename)
        with open(stats_path, 'w') as f:
            json.dump(summary_stats, f, indent=4)
        file_paths['summary_stats'] = stats_path
        
        # Create visualizations
        emit_progress(session_id, "🗺️ Creating interactive map...", socketio)
        
        # Convert to WGS84 for mapping
        nyeri_gdf_wgs84 = nyeri_gdf.to_crs(epsg=4326)
        collection_points_gdf_wgs84 = collection_points_gdf.to_crs(epsg=4326)
        landfill_sites_gdf_wgs84 = landfill_sites_gdf.to_crs(epsg=4326)
        routes_gdf_wgs84 = routes_gdf.to_crs(epsg=4326)

        # Create interactive map
        m = folium.Map(
            location=[nyeri_gdf_wgs84.centroid.y.mean(), nyeri_gdf_wgs84.centroid.x.mean()],
            zoom_start=12,
            tiles='cartodbpositron'
        )

        # Add title
        title_html = '''
        <div style="position: fixed; 
                    top: 10px; left: 50px; width: 300px; height: 30px; 
                    z-index:9999; font-size:20px; font-weight: bold;
                    background-color: rgba(255, 255, 255, 0.8);
                    border-radius: 5px; padding: 5px;
                    font-family: Arial, sans-serif;">
            Waste Collection Network Analysis
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))

        # Add routes to map
        for idx, route in routes_gdf_wgs84.iterrows():
            coords = [(y, x) for x, y in route.geometry.coords]
            distance_km = route['distance_meters'] / 1000
            time_minutes = distance_km * (60 / 40)
            
            popup_html = f"""
            <div style="font-family: Arial, sans-serif;">
                <h4>Route Details</h4>
                <ul style="padding-left: 20px;">
                    <li>From: Collection Point {route['collection_point_id']}</li>
                    <li>To: Landfill {route['landfill_id']}</li>
                    <li>Distance: {distance_km:.2f} km</li>
                    <li>Est. Time: {time_minutes:.2f} min</li>
                </ul>
            </div>
            """
            
            folium.PolyLine(
                coords,
                weight=2,
                color='blue',
                opacity=0.6,
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(m)

        # Add collection points
        for idx, point in collection_points_gdf_wgs84.iterrows():
            popup_html = f"""
            <div style="font-family: Arial, sans-serif;">
                <h4>Collection Point {point['point_id']}</h4>
                <p>{point['description']}</p>
            </div>
            """
            
            folium.CircleMarker(
                location=[point.geometry.y, point.geometry.x],
                radius=6,
                color='red',
                fill=True,
                popup=folium.Popup(popup_html, max_width=200)
            ).add_to(m)

        # Add landfill sites
        for idx, site in landfill_sites_gdf_wgs84.iterrows():
            popup_html = f"""
            <div style="font-family: Arial, sans-serif;">
                <h4>{site['name']}</h4>
                <p>Suitability Score: {site['suitability_score']:.2f}</p>
                <p>Class: {site['suitability_class']}</p>
            </div>
            """
            
            folium.GeoJson(
                site.geometry.__geo_interface__,
                style_function=lambda x: {
                    'fillColor': 'green',
                    'color': 'black',
                    'weight': 2,
                    'fillOpacity': 0.7
                },
                popup=folium.Popup(popup_html, max_width=200)
            ).add_to(m)

        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color: rgba(255, 255, 255, 0.8);
                    border-radius: 5px; padding: 10px;
                    font-family: Arial, sans-serif;">
            <p><i style="background: red; border-radius: 50%; width: 10px; height: 10px; display: inline-block;"></i> Collection Points</p>
            <p><i style="background: green; border-radius: 2px; width: 10px; height: 10px; display: inline-block;"></i> Landfill Sites</p>
            <p><i style="background: blue; width: 10px; height: 2px; display: inline-block;"></i> Routes</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        # Save maps
        latest_map_filename = f'network_analysis_session_{session_id}_latest.html'
        timestamped_map_filename = f'network_analysis_session_{session_id}_{timestamp}.html'
        
        latest_map_path = os.path.join(output_dir, latest_map_filename)
        timestamped_map_path = os.path.join(output_dir, timestamped_map_filename)
        
        m.save(str(latest_map_path))
        m.save(str(timestamped_map_path))
        
        file_paths.update({
            'latest_map': latest_map_path,
            'timestamped_map': timestamped_map_path
        })
        
        # Save routes GeoJSON
        routes_filename = f'routes_session_{session_id}_{timestamp}.geojson'
        routes_path = os.path.join(output_dir, routes_filename)
        routes_gdf.to_file(str(routes_path), driver='GeoJSON')
        file_paths['routes_geojson'] = routes_path

        emit_success(session_id, "🏁 Network analysis complete!", socketio)
        
        # Emit completion event with file paths
        if socketio:
            socketio.emit('network_analysis_complete', {
                'session_id': session_id,
                'files': file_paths,
                'stats': summary_stats
            }, room=session_id)
        
        return {
            'files': file_paths,
            'stats': summary_stats
        }
        
    except Exception as e:
        emit_error(session_id, f"Error in network analysis: {str(e)} 🤯", socketio)
        traceback.print_exc()
        return None