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
        
        # Get dummy data if needed
        if collection_points_gdf is None:
            nyeri_gdf, collection_points_gdf, landfill_sites_gdf = create_dummy_data(session_id, socketio)
        else:
            _, _, landfill_sites_gdf = create_dummy_data(session_id, socketio)

        # Ensure proper CRS
        if not nyeri_gdf.crs:
            nyeri_gdf.set_crs(epsg=4326, inplace=True)
        
        if not collection_points_gdf.crs:
            collection_points_gdf.set_crs(epsg=4326, inplace=True)
        
        # Standardize CRS
        nyeri_gdf = nyeri_gdf.to_crs(epsg=4326)
        collection_points_gdf = collection_points_gdf.to_crs(epsg=4326)
        landfill_sites_gdf = landfill_sites_gdf.to_crs(epsg=4326)

        bounds = nyeri_gdf.total_bounds
        
        # Get road network
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
        except Exception as e:
            emit_error(session_id, f"OSM download failed: {str(e)} 😞", socketio)
            
            if engine:
                G = get_road_network_from_db(engine, bounds, session_id, socketio)
                if not G:
                    raise Exception("Failed to get road network from both OSM and database. ❌")
            else:
                raise Exception("No database connection available for fallback. ❌")

        # Project network
        G = ox.project_graph(G, to_crs='EPSG:21037')
        
        # Project all GeoDataFrames
        nyeri_gdf = nyeri_gdf.to_crs(epsg=21037)
        collection_points_gdf = collection_points_gdf.to_crs(epsg=21037)
        landfill_sites_gdf = landfill_sites_gdf.to_crs(epsg=21037)

        # Get nodes and edges
        nodes, edges = ox.graph_to_gdfs(G)
        
        # Find nearest nodes
        collection_nodes = []
        for idx, point in collection_points_gdf.iterrows():
            nearest = ox.distance.nearest_nodes(G, point.geometry.x, point.geometry.y)
            collection_nodes.append(nearest)
            
        landfill_nodes = []
        for idx, point in landfill_sites_gdf.iterrows():
            nearest = ox.distance.nearest_nodes(G, point.geometry.x, point.geometry.y)
            landfill_nodes.append(nearest)

         # Calculate routes
        routes_data = []
        total_routes = len(landfill_nodes) * len(collection_nodes)  # Changed order to landfill first
        completed_routes = 0
        
        for lf_idx, lf_node in enumerate(landfill_nodes):  # Start with landfill nodes
            for cp_idx, cp_node in enumerate(collection_nodes):  # Then collection points
                try:
                    route = nx.shortest_path(G, lf_node, cp_node, weight='length')  # From landfill to collection point
                    if len(route) < 2:  # Check if route has at least 2 points
                        continue
                        
                    length = nx.shortest_path_length(G, lf_node, cp_node, weight='length')
                    
                    # Get coordinates for the route
                    route_coords = []
                    for node in route:
                        x = G.nodes[node]['x']
                        y = G.nodes[node]['y']
                        if (x, y) not in route_coords:  # Avoid duplicate coordinates
                            route_coords.append((x, y))
                    
                    # Only create LineString if we have at least 2 unique coordinates
                    if len(route_coords) >= 2:
                        routes_data.append({
                            'landfill_id': lf_idx + 1,  # Changed order to match FROM-TO
                            'collection_point_id': cp_idx + 1,
                            'distance_meters': length,
                            'geometry': LineString(route_coords)
                        })
                    
                    completed_routes += 1
                    if completed_routes % 10 == 0:
                        progress = (completed_routes / total_routes) * 100
                        emit_progress(session_id, f"📈 Route calculation progress: {progress:.1f}% 🔄", socketio)
                        
                except nx.NetworkXNoPath:
                    emit_error(session_id, f"No route found between LF {lf_idx + 1} and CP {cp_idx + 1} 🚫", socketio)
                except Exception as e:
                    emit_error(session_id, f"Error calculating route LF {lf_idx + 1} to CP {cp_idx + 1}: {str(e)} 🚫", socketio)
                    continue
        
        # Create routes GeoDataFrame only if we have valid routes
        if routes_data:
            routes_gdf = gpd.GeoDataFrame(routes_data, crs='EPSG:21037')
        else:
            emit_error(session_id, "No valid routes were created", socketio)
            return None
        
        # Create routes GeoDataFrame
        routes_gdf = gpd.GeoDataFrame(routes_data, crs='EPSG:21037')
        
        emit_progress(session_id, "📊 Calculating collection point distances...", socketio)
        routes_summary, summary_stats = calculate_collection_point_distances(routes_gdf, session_id, socketio)
        
        if routes_summary.empty:
            emit_error(session_id, "Failed to calculate route summary", socketio)
            return None
        
        # Save routes summary to CSV
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(output_dir, f'routes_summary_{timestamp}.csv')
        routes_summary.to_csv(summary_path, index=False)
        
        # Save summary statistics to a separate file
        stats_path = os.path.join(output_dir, f'summary_stats_{timestamp}.json')
        with open(stats_path, 'w') as f:
            json.dump(summary_stats, f, indent=4)
        
        # Print formatted table
        print("\nTable: Summary of breakdown of the optimal route.\n")
        print(routes_summary.to_string(index=False))
        
        # Create visualizations
        nyeri_gdf_wgs84 = nyeri_gdf.to_crs(epsg=4326)
        collection_points_gdf_wgs84 = collection_points_gdf.to_crs(epsg=4326)
        landfill_sites_gdf_wgs84 = landfill_sites_gdf.to_crs(epsg=4326)
        routes_gdf_wgs84 = routes_gdf.to_crs(epsg=4326)

        # Create interactive map
        m = folium.Map(
            location=[nyeri_gdf_wgs84.centroid.y.mean(), nyeri_gdf_wgs84.centroid.x.mean()],
            zoom_start=12
        )

        # Add routes to map with enhanced popups
        for idx, route in routes_gdf_wgs84.iterrows():
            coords = [(y, x) for x, y in route.geometry.coords]
            distance_km = route['distance_meters'] / 1000
            time_minutes = distance_km * (60 / 40)
            
            popup_html = f"""
            <div style="font-family: Arial, sans-serif;">
                <h4>Route Details</h4>
                <ul style="padding-left: 20px;">
                    <li>From: C{route['collection_point_id']}</li>
                    <li>To: C{route['landfill_id']}</li>
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

        # Add collection points with enhanced popups
        for idx, point in collection_points_gdf_wgs84.iterrows():
            popup_html = f"""
            <div style="font-family: Arial, sans-serif;">
                <h4>Collection Point {idx + 1}</h4>
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

        # Add landfill sites with enhanced popups
        for idx, point in landfill_sites_gdf_wgs84.iterrows():
            popup_html = f"""
            <div style="font-family: Arial, sans-serif;">
                <h4>{point['name']}</h4>
                <p>Capacity: {point['capacity']} tons</p>
            </div>
            """
            
            folium.CircleMarker(
                location=[point.geometry.y, point.geometry.x],
                radius=8,
                color='green',
                fill=True,
                popup=folium.Popup(popup_html, max_width=200)
            ).add_to(m)

        # Save outputs
        latest_map_path = os.path.join(output_dir, 'network_analysis_latest.html')
        timestamped_map_path = os.path.join(output_dir, f'network_analysis_{timestamp}.html')
        
        m.save(str(latest_map_path))
        m.save(str(timestamped_map_path))
        
        routes_path = os.path.join(output_dir, f'routes_{timestamp}.geojson')
        routes_gdf.to_file(str(routes_path), driver='GeoJSON')

        emit_success(session_id, "🏁 Network analysis complete!", socketio)
        
        return {
            'interactive_map': str(latest_map_path),
            'timestamped_map': str(timestamped_map_path),
            'routes_geojson': str(routes_path),
            'routes_summary': str(summary_path),
            'summary_stats': str(stats_path),
            'stats': summary_stats
        }
        
    except Exception as e:
        emit_error(session_id, f"Error in network analysis: {str(e)} 🤯", socketio)
        traceback.print_exc()
        return None