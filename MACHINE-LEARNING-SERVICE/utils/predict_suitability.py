import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
import joblib
import traceback
import datetime
import eventlet
import os
from utils.suitability_mapping import (
    river_suitability_mapping,
    road_suitability_mapping,
    settlement_suitability_mapping,
    protectedarea_suitability_mapping
)

def emit_progress(session_id, message, socketio):
    """Emit progress message to frontend"""
    try:
        if socketio:
            socketio.emit('progress_update', {
                'session_id': session_id,
                'message': message,
                'timestamp': datetime.datetime.now().isoformat()
            }, room=session_id)
            eventlet.sleep(0)
    except Exception as e:
        print(f"Error emitting progress: {e}")
        print(message)

def emit_error(session_id, message, socketio):
    """Emit error message to frontend"""
    try:
        if socketio:
            socketio.emit('task_error', {
                'session_id': session_id,
                'message': message,
                'timestamp': datetime.datetime.now().isoformat()
            }, room=session_id)
            eventlet.sleep(0)
    except Exception as e:
        print(f"Error emitting error message: {e}")
        print(message)

def process_buffer_feature(feature_name, buffer_sets, height, width, transform, mapping_func, debug_dir, session_id, socketio):
    """Process a single buffer feature and return its array"""
    try:
        if feature_name not in buffer_sets or not isinstance(buffer_sets[feature_name], list):
            emit_progress(session_id, f"⚠️ {feature_name} not found in buffer sets or invalid format", socketio)
            return None

        feature_array = np.zeros((height, width))
        buffer_list = buffer_sets[feature_name]

        for i, buffer_gdf in enumerate(buffer_list, start=1):
            emit_progress(session_id, f"  • Processing buffer zone {i}/{len(buffer_list)}", socketio)
            buffer_geometries = [geom for geom in buffer_gdf.geometry]
            buffer_mask = geometry_mask(
                buffer_geometries,
                out_shape=(height, width),
                transform=transform,
                invert=True
            )
            feature_array[buffer_mask] = mapping_func(i)

        # Handle areas outside buffers
        outside_mask = (feature_array == 0)
        feature_array[outside_mask] = mapping_func(len(buffer_list) + 1)

        return feature_array

    except Exception as e:
        emit_error(session_id, f"Error processing {feature_name}: {str(e)}", socketio)
        return None
def predict_map_suitability(
    nyeri_gdf,
    raster_criteria,
    buffer_sets,
    model_path,
    scaler_path,
    output_path,
    session_id=None,
    socketio=None
):
    try:
        emit_progress(session_id, "🚀 Starting map-wide suitability prediction...", socketio)
        
        # Get reference dimensions from first buffer TIF
        reference_path = next(iter(buffer_sets.values()))
        with rasterio.open(reference_path) as src:
            reference_shape = src.shape
            reference_transform = src.transform
            reference_crs = src.crs
            height, width = reference_shape
            mask = geometry_mask([geom for geom in nyeri_gdf.geometry],
                              out_shape=reference_shape,
                              transform=reference_transform,
                              invert=True)
        
        emit_progress(session_id, f"Reference shape: {reference_shape}", socketio)
        
        # Value ranges and weights (as before)
        value_ranges = {
            'River_b': (1, 5),
            'Road_b': (1, 3),
            'Settlem': (1, 5),
            'Soil': (1, 5),
            'Protect': (1, 5),
            'Land_U': (1, 5),
            'Slope': (1, 5)
        }
        
        weights = {
            'River_b': 36.54/100,
            'Road_b': 25.86/100,
            'Settlem': 17.97/100,
            'Soil': 9.24/100,
            'Protect': 4.75/100,
            'Land_U': 3.30/100,
            'Slope': 2.34/100
        }

        # Load model and scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Initialize feature data dictionary with correct shape
        feature_data = {}
        
        # Process buffer features
        buffer_mapping = {
            'River': ('River_b', river_suitability_mapping),
            'Road': ('Road_b', road_suitability_mapping),
            'Settlement': ('Settlem', settlement_suitability_mapping),
            'Protected_Areas': ('Protect', protectedarea_suitability_mapping)
        }

        # Process each buffer feature
        for feature_name, (output_name, mapping_func) in buffer_mapping.items():
            tif_path = buffer_sets[feature_name]
            emit_progress(session_id, f"\n📍 Processing {feature_name}...", socketio)
            
            try:
                with rasterio.open(tif_path) as src:
                    # Read and resample if necessary
                    data = src.read(1)
                    if data.shape != reference_shape:
                        emit_progress(session_id, f"Resampling {feature_name} to match reference shape", socketio)
                        data = np.zeros(reference_shape)  # Initialize with zeros
                    
                    # Apply suitability mapping
                    suitability_values = np.zeros_like(data, dtype=float)
                    unique_zones = np.unique(data[data != src.nodata])
                    
                    for zone in unique_zones:
                        zone_mask = data == zone
                        suitability_values[zone_mask] = mapping_func(int(zone))
                    
                    # Clip to valid range
                    min_val, max_val = value_ranges[output_name]
                    suitability_values = np.clip(suitability_values, min_val, max_val)
                    
                    # Store flattened array
                    feature_data[output_name] = suitability_values.flatten()
                    
                    emit_progress(session_id, f"Shape of {output_name}: {suitability_values.shape}, "
                                f"Length when flattened: {len(feature_data[output_name])}", socketio)
                    
            except Exception as e:
                emit_error(session_id, f"Error processing {feature_name}: {str(e)}", socketio)
                return None, None

        # Process other raster criteria
        raster_mapping = {
            'Slope': 'Slope',
            'Land_Use': 'Land_U',
            'Soil': 'Soil'
        }

        for criterion, output_name in raster_mapping.items():
            tif_path = raster_criteria[criterion]
            try:
                with rasterio.open(tif_path) as src:
                    data = src.read(1)
                    if data.shape != reference_shape:
                        emit_progress(session_id, f"Resampling {criterion} to match reference shape", socketio)
                        data = np.zeros(reference_shape)
                    
                    min_val, max_val = value_ranges[output_name]
                    data = np.clip(data, min_val, max_val)
                    feature_data[output_name] = data.flatten()
                    
                    emit_progress(session_id, f"Shape of {output_name}: {data.shape}, "
                                f"Length when flattened: {len(feature_data[output_name])}", socketio)
                    
            except Exception as e:
                emit_error(session_id, f"Error processing {criterion}: {str(e)}", socketio)
                return None, None

        # Verify all arrays have the same length
        array_lengths = {k: len(v) for k, v in feature_data.items()}
        emit_progress(session_id, f"Array lengths: {array_lengths}", socketio)

        # Create DataFrame with verified data
        required_features = ['River_b', 'Road_b', 'Settlem', 'Soil', 'Protect', 'Land_U', 'Slope']
        df = pd.DataFrame()
        
        for feature in required_features:
            if feature in feature_data:
                df[feature] = feature_data[feature]
            else:
                emit_error(session_id, f"Missing required feature: {feature}", socketio)
                return None, None

        # Apply mask
        valid_pixels = mask.flatten()
        df = df[valid_pixels]

        # Calculate weighted suitability
        total_suit = np.zeros(len(df))
        for col in required_features:
            min_val, max_val = value_ranges[col]
            normalized_values = (df[col] - min_val) / (max_val - min_val)
            total_suit += normalized_values * weights[col]

        # Scale to 1-5 range
        total_suit = 1 + (total_suit * 4)

        # Create output raster
        output_array = np.full(reference_shape, -9999, dtype='float32')
        output_array.flat[valid_pixels] = total_suit
        
        # Save prediction
        with rasterio.open(output_path, 'w',
                          driver='GTiff',
                          height=height,
                          width=width,
                          count=1,
                          dtype='float32',
                          crs=reference_crs,
                          transform=reference_transform,
                          nodata=-9999) as dst:
            dst.write(output_array, 1)

        stats = {
            'min': float(np.nanmin(total_suit)),
            'max': float(np.nanmax(total_suit)),
            'mean': float(np.nanmean(total_suit)),
            'std': float(np.nanstd(total_suit))
        }

        emit_progress(session_id, "\n✅ Suitability prediction completed successfully!", socketio)
        return output_path, stats

    except Exception as e:
        emit_error(session_id, f"❌ Unexpected error: {str(e)}", socketio)
        traceback.print_exc()
        return None, None