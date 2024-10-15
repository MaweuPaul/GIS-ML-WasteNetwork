def river_suitability_mapping(distance):
    if distance <= 200:
        return 1  # Not suitable
    elif 200 < distance <= 500:
        return 2  # Less suitable
    elif 500 < distance <= 1000:
        return 3  # Moderately Suitable
    elif 1000 < distance <= 1500:
        return 4  # Suitable
    else:
        return 5  # Highly suitable

def residential_area_suitability_mapping(distance):
    if distance <= 200:
        return 1  # Not suitable
    elif 200 < distance <= 500:
        return 2  # Less suitable
    elif 500 < distance <= 1000:
        return 3  # Moderately Suitable
    elif 1000 < distance <= 1500:
        return 4  # Suitable
    else:
        return 5  # Highly suitable

def soil_suitability_mapping(soil_type):
    soil_type = soil_type.lower()
    if soil_type == 'sand':
        return 1  # Not suitable
    elif soil_type == 'loam':
        return 2  # Less suitable
    elif soil_type == 'silt':
        return 4  # Suitable
    elif soil_type == 'clay':
        return 5  # Highly suitable
    else:
        return 0  # Unknown soil type

def road_suitability_mapping(distance):
    if distance <= 200:
        return 1  # Not suitable
    elif 200 < distance <= 500:
        return 3  # Moderately suitable
    elif 500 < distance <= 1000:
        return 5  # Highly Suitable
    elif 1000 < distance <= 1500:
        return 4  # Suitable
    else:
        return 2  # Less suitable

def settlement_suitability_mapping(distance):
    if distance <= 200:
        return 1  # Not suitable
    elif 200 < distance <= 500:
        return 2  # Less suitable
    elif 500 < distance <= 1000:
        return 3  # Moderately Suitable
    elif 1000 < distance <= 1500:
        return 4  # Suitable
    else:
        return 5  # Highly suitable

def protected_areas_suitability_mapping(distance):
    if distance <= 200:
        return 1  # Not suitable
    elif 200 < distance <= 500:
        return 2  # Less suitable
    elif 500 < distance <= 1000:
        return 3  # Moderately Suitable
    elif 1000 < distance <= 1500:
        return 4  # Suitable
    else:
        return 5  # Highly suitable

def geology_suitability_mapping(geology_type):
    # You haven't provided specific criteria for geology
    # So I'm leaving this as a placeholder function
    # You should implement the logic based on your geology classification
    return 3  # Default to moderately suitable

def landuse_suitability_mapping(land_use):
    land_use = land_use.lower()
    if land_use in ['built up', 'forests']:
        return 1  # Not suitable
    elif land_use == 'farmlands':
        return 2  # Less suitable
    elif land_use == 'bareland':
        return 5  # Highly suitable
    else:
        return 0  # Unknown land use type

# Additional function to map slope suitability
def slope_suitability_mapping(slope_degree):
    if slope_degree <= 5:
        return 5  # Highly suitable (0-5 degrees)
    elif 5 < slope_degree <= 10:
        return 4  # Suitable (5-10 degrees)
    elif 10 < slope_degree <= 15:
        return 3  # Moderately suitable (10-15 degrees)
    elif 15 < slope_degree <= 20:
        return 2  # Less suitable (15-20 degrees)
    else:
        return 1  # Not suitable (>20 degrees)