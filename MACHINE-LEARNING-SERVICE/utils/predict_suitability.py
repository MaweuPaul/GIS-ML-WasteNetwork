import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import rasterio
import os
import traceback
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patheffects as pe
import datetime
import eventlet
import joblib
from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from geoalchemy2 import Geometry
from geoalchemy2.shape import from_shape
from shapely.geometry import Point
from shapely.ops import unary_union
from sqlalchemy import func


Base = declarative_base()
from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from geoalchemy2 import Geometry
from geoalchemy2.shape import from_shape
from shapely.geometry import Point

Base = declarative_base()

class CollectionPoint(Base):
    __tablename__ = 'collection_points'

    id = Column(Integer, primary_key=True)

    # Define relationship
    routes = relationship("Route", back_populates="collection_point")

class Route(Base):
    __tablename__ = 'routes'

    id = Column(Integer, primary_key=True, autoincrement=True)
    collection_point_id = Column(Integer, ForeignKey('collection_points.id'), nullable=False)
    landfill_site_id = Column(Integer, ForeignKey('landfill_sites.id'), nullable=False)
    distance_meters = Column(Float, nullable=False)
    geom = Column(Geometry(geometry_type='LINESTRING', srid=21037), nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Use string reference for relationship
    collection_point = relationship("CollectionPoint", back_populates="routes")
    landfill_site = relationship("LandfillSite", back_populates="routes")

class LandfillSite(Base):
    __tablename__ = 'landfill_sites'

    id = Column(Integer, primary_key=True)
    landfill_id = Column(Integer, unique=True, nullable=False)
    suitability_score = Column(Float, nullable=False)
    suitability_class = Column(String, nullable=False)
    geom = Column(Geometry(geometry_type='GEOMETRY', srid=21037), nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Define relationship
    routes = relationship("Route", back_populates="landfill_site")
def emit_progress(session_id, message, socketio):
    try:
        if socketio:
            socketio.emit('progress_update', {'session_id': session_id, 'message': message}, room=session_id)
            print(f"Emitted progress: {message}")
            eventlet.sleep(0)
        else:
            print("Progress message:", message)
    except Exception as e:
        print(f"Failed to emit progress message: {e}")

def emit_error(session_id, message, socketio):
    try:
        if socketio:
            socketio.emit('task_error', {'session_id': session_id, 'message': message}, room=session_id)
            print(f"Emitted error: {message}")
            eventlet.sleep(0)
        else:
            print("Error message:", message)
    except Exception as e:
        print(f"Failed to emit error message: {e}")

def save_continuous_sites_to_db(continuous_sites_gdf, engine, session_id=None, socketio=None):
    try:
        emit_progress(session_id, "\n🔄 Starting database export process...", socketio)
        emit_progress(session_id, f"📊 Total sites to process: {len(continuous_sites_gdf)}", socketio)
        
        session = Session(engine)
        sites_exported = 0
        sites_created = 0
        
        try:
            emit_progress(session_id, "\n🗑️ Deleting existing records...", socketio)
            
            # Correct the class name from Routes to Route
            session.query(Route).filter(Route.landfill_site_id.in_(
                session.query(LandfillSite.id)
            )).delete(synchronize_session=False)
            session.commit()
            
            # Now delete records in the landfill_sites table
            deleted_count = session.query(LandfillSite).delete()
            session.commit()
            emit_progress(session_id, f"  ✅ Deleted {deleted_count} existing records", socketio)
            
            emit_progress(session_id, "\n📥 Processing new sites...", socketio)
            current_time = func.now()
            
            for idx, row in continuous_sites_gdf.iterrows():
                try:
                    site = LandfillSite(
                        landfill_id=int(row['site_id']),
                        suitability_score=float(row['Total_Suit']),
                        suitability_class=str(row['Suitability_Class']),
                        geom=from_shape(row.geometry, srid=21037),
                        created_at=current_time,
                        updated_at=current_time
                    )
                    session.add(site)
                    sites_created += 1
                    session.commit()
                    sites_exported += 1
                    emit_progress(
                        session_id,
                        f"  ✅ Successfully created site {row['site_id']} "
                        f"(Score: {row['Total_Suit']:.2f}, Class: {row['Suitability_Class']})",
                        socketio
                    )
                except Exception as e:
                    session.rollback()
                    emit_error(
                        session_id,
                        f"❌ Error processing site {row['site_id']}: {str(e)}",
                        socketio
                    )
                    continue
            
            emit_progress(session_id, "\n📊 Export Summary:", socketio)
            emit_progress(session_id, f"  • Previous records deleted: {deleted_count}", socketio)
            emit_progress(session_id, f"  • New sites created: {sites_created}", socketio)
            emit_progress(session_id, "✨ Database export completed successfully!", socketio)
            
            return {
                'deleted_count': deleted_count,
                'sites_created': sites_created
            }
            
        finally:
            session.close()
            emit_progress(session_id, "🔒 Database session closed", socketio)
        
    except Exception as e:
        emit_error(session_id, f"❌ Database export error: {str(e)}", socketio)
        traceback.print_exc()
        return None
    
def classify_suitability(score):
    if score <= 2.0:
        return 'Not Suitable'
    elif score <= 2.75:
        return 'Less Suitable'
    elif score <= 3.5:
        return 'Moderately Suitable'
    elif score <= 4.25:
        return 'Suitable'
    else:
        return 'Highly Suitable'

def save_continuous_sites(continuous_sites_gdf, output_dir, session_id, socketio=None):
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        export_gdf = continuous_sites_gdf.copy()
        export_gdf['area_ha'] = export_gdf['area_m2'] / 10000
        numeric_columns = ['Total_Suit', 'area_ha']
        for col in numeric_columns:
            if col in export_gdf.columns:
                export_gdf[col] = export_gdf[col].round(2)
        
        geojson_path = os.path.join(output_dir, f'continuous_sites_{session_id}.geojson')
        export_gdf.to_file(geojson_path, driver='GeoJSON')
        
        shp_path = os.path.join(output_dir, f'continuous_sites_{session_id}.shp')
        export_gdf.to_file(shp_path)
        
        csv_path = os.path.join(output_dir, f'continuous_sites_{session_id}.csv')
        export_gdf['centroid_lon'] = export_gdf.geometry.centroid.x
        export_gdf['centroid_lat'] = export_gdf.geometry.centroid.y
        export_gdf.drop(columns=['geometry']).to_csv(csv_path, index=False)
        
        emit_progress(session_id, "\n💾 Saving continuous sites...", socketio)
        emit_progress(session_id, f"Saved {len(export_gdf)} continuous sites:", socketio)
        emit_progress(session_id, f"  • GeoJSON: {geojson_path}", socketio)
        emit_progress(session_id, f"  • Shapefile: {shp_path}", socketio)
        emit_progress(session_id, f"  • CSV: {csv_path}", socketio)
        
        return {
            'geojson_path': geojson_path,
            'shapefile_path': shp_path,
            'csv_path': csv_path,
            'site_count': len(export_gdf)
        }
        
    except Exception as e:
        emit_error(session_id, f"Error saving continuous sites: {str(e)}", socketio)
        return None

def save_landfill_locations(candidate_gdf, output_dir, session_id, socketio=None):
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        landfill_locations = gpd.GeoDataFrame({
            'landfill_id': range(1, len(candidate_gdf) + 1),
            'suitability_score': candidate_gdf['Total_Suit'],
            'suitability_class': candidate_gdf['Suitability_Class'],
            'geometry': candidate_gdf.geometry
        }, crs=candidate_gdf.crs)
        
        landfill_locations['longitude'] = landfill_locations.geometry.centroid.x
        landfill_locations['latitude'] = landfill_locations.geometry.centroid.y
        
        geojson_path = os.path.join(output_dir, f'landfill_locations_{session_id}.geojson')
        shp_path = os.path.join(output_dir, f'landfill_locations_{session_id}.shp')
        csv_path = os.path.join(output_dir, f'landfill_locations_{session_id}.csv')
        
        landfill_locations.to_file(geojson_path, driver='GeoJSON')
        landfill_locations.to_file(shp_path)
        landfill_locations.drop(columns=['geometry']).to_csv(csv_path, index=False)
        
        emit_progress(session_id, "\n💾 Saving landfill locations...", socketio)
        emit_progress(session_id, f"Saved {len(landfill_locations)} landfill locations:", socketio)
        emit_progress(session_id, f"  • GeoJSON: {geojson_path}", socketio)
        emit_progress(session_id, f"  • Shapefile: {shp_path}", socketio)
        emit_progress(session_id, f"  • CSV: {csv_path}", socketio)
        
        return {
            'geojson_path': geojson_path,
            'shapefile_path': shp_path,
            'csv_path': csv_path,
            'landfill_count': len(landfill_locations)
        }
        
    except Exception as e:
        emit_error(session_id, f"Error saving landfill locations: {str(e)}", socketio)
        return None

def clean_data(points_gdf):
    points_gdf = points_gdf.copy()
    required_columns = ['River_b', 'Road_b', 'Protect', 'Settlem', 'Slope', 'Land_U', 'Soil']
    
    for col in required_columns:
        if col not in points_gdf.columns:
            points_gdf[col] = np.nan
    
    for col in required_columns:
        points_gdf[col] = pd.to_numeric(points_gdf[col], errors='coerce')
    
    if 'Land_U' in points_gdf.columns:
        points_gdf['Land_U'] = points_gdf['Land_U'].fillna(0)
    else:
        points_gdf['Land_U'] = 0
    
    columns_to_interpolate = ['River_b', 'Road_b', 'Protect', 'Settlem', 'Slope', 'Soil']
    
    for col in columns_to_interpolate:
        if col in points_gdf.columns:
            col_mean = points_gdf[col].mean()
            if pd.isna(col_mean):
                col_mean = 3
            
            points_gdf[col] = points_gdf[col].interpolate(method='nearest', limit_direction='both')
            points_gdf[col] = points_gdf[col].fillna(col_mean)
            points_gdf[col] = points_gdf[col].round(0)
            
            if col == 'Protect':
                points_gdf[col] = points_gdf[col].clip(1, 5)
            elif col in ['River_b', 'Road_b', 'Settlem', 'Soil']:
                points_gdf[col] = points_gdf[col].clip(1, 5)
            elif col == 'Slope':
                points_gdf[col] = points_gdf[col].clip(0, 4)
    
    return points_gdf

def create_suitability_map(gdf, nyeri_gdf, title, output_path):
    color_dict = {
        'Not Suitable': '#d7191c',
        'Less Suitable': '#fdae61',
        'Moderately Suitable': '#ffffbf',
        'Suitable': '#a6d96a',
        'Highly Suitable': '#1a9641'
    }
    
    fig, ax = plt.subplots(figsize=(15, 10))
    nyeri_gdf.plot(ax=ax, alpha=0.3, color='lightgray')
    
    for suitability_class, color in color_dict.items():
        mask = gdf['Suitability_Class'] == suitability_class
        class_gdf = gdf[mask]
        if not class_gdf.empty:
            class_gdf.plot(
                ax=ax,
                color=color,
                label=suitability_class,
                alpha=0.7
            )
    
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles, 
        labels,
        title='Suitability Classes',
        title_fontsize=12,
        fontsize=10,
        loc='upper right',
        frameon=True,
        facecolor='white',
        edgecolor='black'
    )
    
    ax.set_title(title, fontsize=16, pad=20)
    ax.annotate('N', xy=(0.98, 0.98), xycoords='axes fraction',
               horizontalalignment='center', verticalalignment='center',
               fontsize=18, fontweight='bold',
               path_effects=[pe.withStroke(linewidth=3, foreground="w")])
    
    scalebar = ScaleBar(1, 'km', dimension='si-length', location='lower right')
    ax.add_artist(scalebar)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_candidate_map(gdf, nyeri_gdf, title, output_path):
    fig, ax = plt.subplots(figsize=(15, 10))
    nyeri_gdf.plot(ax=ax, alpha=0.3, color='lightgray')
    
    gdf.plot(
        ax=ax,
        color='#90EE90',
        label='Suitable',
        alpha=0.7
    )
    
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles, 
        labels,
        title='Suitability Classes',
        title_fontsize=12,
        fontsize=10,
        loc='upper right',
        frameon=True,
        facecolor='white',
        edgecolor='black'
    )
    
    ax.set_title(title, fontsize=16, pad=20)
    ax.annotate('N', xy=(0.98, 0.98), xycoords='axes fraction',
               horizontalalignment='center', verticalalignment='center',
               fontsize=18, fontweight='bold',
               path_effects=[pe.withStroke(linewidth=3, foreground="w")])
    
    scalebar = ScaleBar(1, 'km', dimension='si-length', location='lower right')
    ax.add_artist(scalebar)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_continuous_candidate_sites(candidate_gdf, buffer_distance=50, min_area=10000, socketio=None, session_id=None):
    try:
        emit_progress(session_id, "Creating continuous candidate areas...", socketio)
        
        buffered = candidate_gdf.copy()
        buffered.geometry = buffered.geometry.buffer(buffer_distance)
        
        buffered['cluster_id'] = 1
        buffered['mean_suit'] = buffered['Total_Suit']
        
        dissolved = buffered.dissolve(
            by='cluster_id',
            aggfunc={
                'Total_Suit': 'mean',
                'mean_suit': 'mean'
            }
        ).reset_index()
        
        continuous_sites = dissolved.explode(index_parts=True).reset_index(drop=True)
        continuous_sites['area_m2'] = continuous_sites.geometry.area
        continuous_sites = continuous_sites[continuous_sites['area_m2'] >= min_area]
        
        continuous_sites.geometry = continuous_sites.geometry.buffer(0)
        
        if len(continuous_sites) > 0:
            continuous_sites['site_id'] = range(1, len(continuous_sites) + 1)
            continuous_sites['centroid_x'] = continuous_sites.geometry.centroid.x
            continuous_sites['centroid_y'] = continuous_sites.geometry.centroid.y
            continuous_sites['Suitability_Class'] = continuous_sites['Total_Suit'].apply(classify_suitability)
            
            emit_progress(session_id, f"Created {len(continuous_sites)} continuous sites", socketio)
            return continuous_sites
        else:
            emit_progress(session_id, "No continuous sites met the minimum area requirement", socketio)
            return None
        
    except Exception as e:
        emit_error(session_id, f"Error creating continuous sites: {str(e)}", socketio)
        traceback.print_exc()
        return None

def create_continuous_candidate_map(continuous_sites_gdf, nyeri_gdf, title, output_path):
    fig, ax = plt.subplots(figsize=(15, 10))
    nyeri_gdf.plot(ax=ax, alpha=0.3, color='lightgray')
    
    continuous_sites_gdf.plot(
        ax=ax,
        color='#90EE90',
        alpha=0.7,
        label='Suitable Areas'
    )
    
    ax.legend(
        title='Candidate Sites',
        title_fontsize=12,
        fontsize=10,
        loc='upper right',
        frameon=True,
        facecolor='white',
        edgecolor='black'
    )
    
    ax.set_title(title, fontsize=16, pad=20)
    ax.annotate('N', xy=(0.98, 0.98), xycoords='axes fraction',
               horizontalalignment='center', verticalalignment='center',
               fontsize=18, fontweight='bold',
               path_effects=[pe.withStroke(linewidth=3, foreground="w")])
    
    scalebar = ScaleBar(1, 'km', dimension='si-length', location='lower right')
    ax.add_artist(scalebar)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def predict_map_suitability(nyeri_gdf, buffer_sets, raster_criteria, model_path, scaler_path, interval=50, session_id=None, socketio=None, engine=None):
    try:
        stats = {
            'full_map': {},
            'candidate_sites': {},
            'continuous_sites': {}
        }
               
        emit_progress(session_id, "🚀 Starting suitability prediction...", socketio)
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        bounds = nyeri_gdf.total_bounds
        x_coords = np.arange(bounds[0], bounds[2], interval)
        y_coords = np.arange(bounds[1], bounds[3], interval)
        study_area_polygon = nyeri_gdf.unary_union
        
        points = []
        point_data = []
        progress_count = 0
        progress_interval = max(1, len(x_coords) * len(y_coords) // 20)
        
        emit_progress(session_id, "\n📍 Generating grid points...", socketio)
        
        for x in x_coords:
            for y in y_coords:
                point = Point(x, y)
                if study_area_polygon.contains(point):
                    point_info = {
                        'geometry': point,
                        'x': x,
                        'y': y
                    }
                    
                    for feature_name, buffer_list in buffer_sets.items():
                        if not isinstance(buffer_list, list):
                            continue
                        
                        for i, buffer_gdf in enumerate(buffer_list):
                            if any(buffer_gdf.geometry.contains(point)):
                                point_info[f'{feature_name}_buffer'] = i + 1
                                break
                        else:
                            point_info[f'{feature_name}_buffer'] = len(buffer_list) + 1
                    
                    points.append(point)
                    point_data.append(point_info)
                
                progress_count += 1
                if progress_count % progress_interval == 0:
                    progress_percent = (progress_count / (len(x_coords) * len(y_coords))) * 100
                    emit_progress(session_id, f"⏳ Processed {progress_count} points ({progress_percent:.1f}%)", socketio)
        
        points_gdf = gpd.GeoDataFrame(point_data, geometry='geometry', crs=nyeri_gdf.crs)
        
        emit_progress(session_id, "\n🔍 Sampling raster criteria values...", socketio)
        for raster_name, raster_path in raster_criteria.items():
            try:
                with rasterio.open(raster_path) as src:
                    coords = [(p.x, p.y) for p in points_gdf.geometry]
                    values = [val[0] for val in src.sample(coords)]
                    points_gdf[raster_name] = values
            except Exception as e:
                emit_error(session_id, f"⚠️ Error processing {raster_name}: {str(e)}", socketio)
                points_gdf[raster_name] = 3
        
        column_mapping = {
            'River_buffer': 'River_b',
            'Road_buffer': 'Road_b',
            'Protected_Areas_buffer': 'Protect',
            'Settlement_buffer': 'Settlem',
            'landuse': 'Land_U',
            'land_use': 'Land_U',
            'slope': 'Slope'
        }
        points_gdf = points_gdf.rename(columns=column_mapping)
        
        emit_progress(session_id, "\n🧹 Cleaning and interpolating data...", socketio)
        points_gdf = clean_data(points_gdf)
        
        emit_progress(session_id, "\n🤖 Making predictions...", socketio)
        feature_columns = ['River_b', 'Road_b', 'Settlem', 'Soil', 'Protect', 'Land_U', 'Slope']
        X = points_gdf[feature_columns].values
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        points_gdf['Total_Suit'] = predictions.round(2)
        
        points_gdf['Suitability_Class'] = points_gdf['Total_Suit'].apply(classify_suitability)
        
        output_dir = os.path.join(os.getcwd(), 'output')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        emit_progress(session_id, "\n🎨 Creating full suitability map...", socketio)
        full_map_path = os.path.join(output_dir, f"suitability_map_{session_id}.png")
        create_suitability_map(
            points_gdf,
            nyeri_gdf,
            'Landfill Suitability Map',
            full_map_path
        )
        
        emit_progress(session_id, "\n🎯 Creating candidate sites map...", socketio)
        candidate_gdf = points_gdf[points_gdf['Total_Suit'] >= 3.5].copy()
        candidate_map_path = os.path.join(output_dir, f"candidate_sites_map_{session_id}.png")
        create_candidate_map(
            candidate_gdf,
            nyeri_gdf,
            'Candidate Landfill Sites\n(Suitability ≥ 3.5)',
            candidate_map_path
        )
        
        emit_progress(session_id, "\n🔄 Creating continuous candidate areas...", socketio)
        continuous_sites = create_continuous_candidate_sites(
            candidate_gdf,
            buffer_distance=50,
            min_area=5000,
            socketio=socketio,
            session_id=session_id
        )
        
        continuous_map_path = None
        continuous_geojson = None
        
        if continuous_sites is not None and not continuous_sites.empty:
            continuous_map_path = os.path.join(output_dir, f"continuous_candidate_sites_{session_id}.png")
            create_continuous_candidate_map(
                continuous_sites,
                nyeri_gdf,
                'Continuous Candidate Landfill Sites',
                continuous_map_path
            )
            
            if engine:
                emit_progress(session_id, "\n💾 Saving to database...", socketio)
                db_results = save_continuous_sites_to_db(
                    continuous_sites,
                    engine,
                    session_id,
                    socketio
                )
                if db_results:
                    stats['database'] = {
                        'deleted_count': db_results['deleted_count'],
                        'sites_created': db_results['sites_created']
                    }
            
            continuous_files = save_continuous_sites(
                continuous_sites,
                output_dir,
                session_id,
                socketio
            )
            
            if continuous_files:
                continuous_geojson = continuous_files['geojson_path']
                continuous_shapefile = continuous_files['shapefile_path']
                stats['continuous_sites'] = {
                    'total_sites': continuous_files['site_count'],
                    'total_area_ha': continuous_sites['area_m2'].sum() / 10000,
                    'min_area_ha': continuous_sites['area_m2'].min() / 10000,
                    'max_area_ha': continuous_sites['area_m2'].max() / 10000,
                    'mean_area_ha': continuous_sites['area_m2'].mean() / 10000
                }
        
            stats['continuous_sites'] = {
                'total_sites': len(continuous_sites),
                'total_area_ha': continuous_sites['area_m2'].sum() / 10000,
                'min_area_ha': continuous_sites['area_m2'].min() / 10000,
                'max_area_ha': continuous_sites['area_m2'].max() / 10000,
                'mean_area_ha': continuous_sites['area_m2'].mean() / 10000
            }
        else:
            emit_progress(session_id, "⚠️ Could not create continuous sites", socketio)
        
        full_csv_path = os.path.join(output_dir, f"full_suitability_{session_id}.csv")
        candidate_csv_path = os.path.join(output_dir, f"candidate_sites_{session_id}.csv")
        points_gdf.to_csv(full_csv_path, index=False)
        candidate_gdf.to_csv(candidate_csv_path, index=False)
        
        stats = {
            'full_map': {
                'total_points': len(points_gdf),
                'min_score': points_gdf['Total_Suit'].min(),
                'max_score': points_gdf['Total_Suit'].max(),
                'mean_score': points_gdf['Total_Suit'].mean(),
                'class_distribution': points_gdf['Suitability_Class'].value_counts().to_dict()
            },
            'candidate_sites': {
                'total_points': len(candidate_gdf),
                'min_score': candidate_gdf['Total_Suit'].min(),
                'max_score': candidate_gdf['Total_Suit'].max(),
                'mean_score': candidate_gdf['Total_Suit'].mean(),
                'class_distribution': candidate_gdf['Suitability_Class'].value_counts().to_dict()
            }
        }
        
        emit_progress(session_id, "\n📊 Analysis Results:", socketio)
        emit_progress(session_id, f"\nFull Map Statistics:", socketio)
        for class_name, count in stats['full_map']['class_distribution'].items():
            percentage = (count / stats['full_map']['total_points']) * 100
            emit_progress(session_id, f"  • {class_name}: {count} points ({percentage:.1f}%)", socketio)
        
        emit_progress(session_id, f"\nCandidate Sites Statistics:", socketio)
        for class_name, count in stats['candidate_sites']['class_distribution'].items():
            percentage = (count / stats['candidate_sites']['total_points']) * 100
            emit_progress(session_id, f"  • {class_name}: {count} points ({percentage:.1f}%)", socketio)
        
        emit_progress(session_id, f"\n✅ Analysis complete!", socketio)
        emit_progress(session_id, f"Full suitability map saved to: {full_map_path}", socketio)
        emit_progress(session_id, f"Candidate sites map saved to: {candidate_map_path}", socketio)
        
        landfill_paths = save_landfill_locations(candidate_gdf, output_dir, session_id, socketio)
        
        return {
            'full_map_path': full_map_path,
            'candidate_map_path': candidate_map_path,
            'continuous_map_path': continuous_map_path,
            'continuous_geojson': continuous_geojson,
            'continuous_shapefile': continuous_shapefile if 'continuous_shapefile' in locals() else None,
            'full_csv_path': full_csv_path,
            'candidate_csv_path': candidate_csv_path,
            'stats': stats
        }

    except Exception as e:
        emit_error(session_id, f"Error in predict_suitability: {str(e)}", socketio)
        traceback.print_exc()
        return None