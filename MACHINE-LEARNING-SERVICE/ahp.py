# backend/MACHINE-LEARNING-SERVICE/ahp.py
import ahpy

def calculate_weights():
    comparisons = {
        ('Proximity to Roads', 'Proximity to Rivers'): 3,
        ('Proximity to Roads', 'Soil Type'): 5,
        ('Proximity to Roads', 'Geology Type'): 3,
        ('Proximity to Roads', 'Elevation'): 4,
        ('Proximity to Rivers', 'Soil Type'): 1/3,
        ('Proximity to Rivers', 'Geology Type'): 3,
        ('Proximity to Rivers', 'Elevation'): 2,
        ('Soil Type', 'Geology Type'): 5,
        ('Soil Type', 'Elevation'): 7,
        ('Geology Type', 'Elevation'): 3,
    }

    criteria = ahpy.Compare('Criteria', comparisons=comparisons, precision=3)
    weights = criteria.weights
    return weights

 # backend/MACHINE-LEARNING-SERVICE/ahp.py (continued)
def get_weights():
    return calculate_weights()