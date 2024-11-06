import numpy as np
import pandas as pd

def river_suitability_mapping(distance):
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

def road_suitability_mapping(distance):
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

def settlement_suitability_mapping(distance):
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

def protectedarea_suitability_mapping(distance):
    if pd.isna(distance):
        return np.nan
    if distance <= 300:
        return 1  # Not suitable
    elif 300 < distance <= 1000:
        return 2  # Less suitable
    elif 1000 < distance <= 1500:  # Fixed the gap
        return 3  # Moderately suitable
    elif 1500 < distance <= 2000:
        return 4  # Suitable
    else:
        return 5  # Highly suitable

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

def land_use_suitability_mapping(land_use):
    if pd.isna(land_use) or land_use is None:
        return np.nan  # No data
    if 'forests' in land_use or 'settlements' in land_use:
        return 1  # Not suitable
    elif 'farmlands' in land_use:
        return 4  # Suitable
    elif 'bareland' in land_use:
        return 5  # Highly suitable
    else:
        return 2  # Less suitable