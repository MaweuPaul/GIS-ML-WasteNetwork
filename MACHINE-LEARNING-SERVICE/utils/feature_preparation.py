import geopandas as gpd
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import logging
from sqlalchemy import create_engine
import rasterio
from rasterio.features import rasterize
import os

class FeaturePreparation:
    def __init__(self, engine, study_area_boundary, precalculated_data=None):
        """
        Initialize feature preparation with database connection and study area
        
        Args:
            engine: SQLAlchemy database engine
            study_area_boundary: GeoDataFrame of the study area boundary
            precalculated_data: Dict containing pre-calculated data for 'Soil', 'Land_Use', and 'Slope'
        """
        self.engine = engine
        self.study_area = study_area_boundary
        self.precalculated_data = precalculated_data or {}
        self.logger = logging.getLogger(__name__)
        
        # Define criteria weights
        self.criteria_weights = {
        'River': 36.54,
        'Road': 25.86,
        'Settlement': 17.97,
        'Soil': 9.24,
        'Protected Areas': 4.75,
        'Land Use': 3.30,
        'Slope': 2.34
        }

    def river_suitability_mapping(self, distance):
        if pd.isna(distance):
            return np.nan
        if distance <= 300:
            return 1  # Not suitable
        elif 300 < distance <= 1000:
            return 2  # Less suitable
        elif 1000 < distance <= 1500:
            return 3  # Moderately suitable
        elif 1500 < distance <= 2000:
            return 4  # Suitable
        else:
            return 5  # Highly suitable

    def road_suitability_mapping(self, distance):
        if pd.isna(distance):
            return np.nan
        if distance > 1200:
            return 1  # Not suitable
        elif 1000 < distance <= 1200:
            return 2  # Less suitable
        elif 800 < distance <= 1000:
            return 3  # Moderately suitable
        elif 400 < distance <= 800:
            return 4  # Suitable
        else:
            return 5  # Highly suitable

    def settlement_suitability_mapping(self, distance):
        if pd.isna(distance):
            return np.nan
        if distance <= 400:
            return 1  # Not suitable
        elif 400 < distance <= 900:
            return 2  # Less suitable
        elif 1500 < distance <= 2100:
            return 3  # Moderately suitable
        elif distance > 2100:
            return 4  # Suitable
        else:
            return 5  # Highly suitable

    def protectedarea_suitability_mapping(self, distance):
        if pd.isna(distance):
            return np.nan
        if distance <= 300:
            return 1  # Not suitable
        elif 300 < distance <= 1000:
            return 2  # Less suitable
        elif 1500 < distance <= 2000:
            return 3  # Moderately suitable
        elif 2000 < distance <= 2500:
            return 4  # Suitable
        else:
            return 5  # Highly suitable

    @staticmethod
    def slope_suitability_mapping(slope_degree):
        if pd.isna(slope_degree):
            return np.nan
        if slope_degree > 18.6:
            return 1  # Not suitable
        elif 5.3 < slope_degree <= 18.6:
            return 2  # Less suitable
        elif 3.1 < slope_degree <= 5.2:
            return 3  # Moderately suitable
        elif 1.7 < slope_degree <= 3:
            return 4  # Suitable
        else:
            return 5  # Highly suitable

    @staticmethod
    def land_use_suitability_mapping(land_use):
        if pd.isna(land_use) or land_use is None:
            return np.nan
        if 'forests' in land_use or 'settlements' in land_use:
            return 1  # Not suitable
        elif 'farmlands' in land_use:
            return 4  # Suitable
        elif 'bareland' in land_use:
            return 5  # Highly suitable
        else:
            return 2  # Less suitable

    @staticmethod
    def soil_suitability_mapping(soil_type):
        try:
            if pd.isna(soil_type) or soil_type is None:
                return np.nan
            
            soil_type = str(soil_type).strip()
            soil_mappings = {
                'Bk31-2a': 2,  # Calcic Cambisols - Less suitable
                'Ne12-2c': 3,  # Eutric Nitosols - Moderately suitable
                'Nh2-2c': 2    # Humic Nitosols - Less suitable
            }
            
            if soil_type in soil_mappings:
                return soil_mappings[soil_type]
            
            if soil_type.startswith('Bk'):
                return 2
            elif soil_type.startswith('Ne'):
                return 3
            elif soil_type.startswith('Nh'):
                return 2
            
            return 3
            
        except Exception as e:
            print(f"Error in soil_suitability_mapping: {str(e)}")
            return np.nan

    def prepare_features(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare all features for analysis
        """
        try:
            features = pd.DataFrame()
            
            # Process distance-based features
            distance_features = ['River', 'Road', 'Settlement', 'ProtectedArea']
            for feature in distance_features:
                query = f'SELECT geom FROM "{feature}"'
                gdf = gpd.read_postgis(query, self.engine, geom_col='geom')
                
                if not gdf.empty:
                    distances = self.calculate_distances(gdf)
                    suitability_func = getattr(self, f"{feature.lower()}_suitability_mapping")
                    features[feature] = np.vectorize(suitability_func)(distances)
            
            # Add pre-calculated features
            if self.precalculated_data:
                for feature_name, data in self.precalculated_data.items():
                    if data is not None:
                        features[feature_name] = data
            
            # Calculate weighted scores
            for feature_type, weight in self.criteria_weights.items():
                if feature_type in features.columns:
                    features[f'{feature_type}_weighted'] = features[feature_type] * (weight / 100)
            
            # Calculate total suitability score
            features['total_suitability'] = sum(
                features[f'{feature_type}_weighted']
                for feature_type in self.criteria_weights.keys()
                if f'{feature_type}_weighted' in features.columns
            )
            
            feature_columns = features.columns.tolist()
            
            return features, feature_columns
            
        except Exception as e:
            self.logger.error(f"Error in prepare_features: {str(e)}")
            raise

    def calculate_distances(self, gdf: gpd.GeoDataFrame) -> np.ndarray:
        """Calculate distances from study area to nearest features"""
        try:
            distances = self.study_area.geometry.apply(lambda x: gdf.distance(x).min()).values
            return distances
        except Exception as e:
            self.logger.error(f"Error calculating distances: {str(e)}")
            raise

    def save_features(self, features: pd.DataFrame, output_path: str):
        """Save prepared features to file"""
        try:
            features.to_csv(output_path, index=False)
            self.logger.info(f"Features saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving features: {str(e)}")
            raise