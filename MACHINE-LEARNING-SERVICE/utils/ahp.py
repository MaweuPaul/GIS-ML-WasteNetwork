import numpy as np
from tabulate import tabulate

def calculate_weights():
    """
    Calculate weights for landfill site suitability criteria using AHP with Saaty's scale.
    """
    # Define the criteria based on the provided datasets
    criteria = [
        'River',
        'Road',
        'Settlement',
        'Soil',
        'Protected Areas',
        'Land Use',
        'Slope'
    ]
    n = len(criteria)
    
    # Pairwise comparison matrix using Saaty's scale (1-9)
    matrix = np.array([
        [1,    2,    3,    5,    7,    8,    9],    # River
        [1/2,  1,    2,    4,    6,    7,    8],    # Road
        [1/3,  1/2,  1,    3,    5,    6,    7],    # Settlement
        [1/5,  1/4,  1/3,  1,    3,    4,    5],    # Soil
        [1/7,  1/6,  1/5,  1/3,  1,    2,    3],    # Protected Areas
        [1/8,  1/7,  1/6,  1/4,  1/2,  1,    2],    # Land Use
        [1/9,  1/8,  1/7,  1/5,  1/3,  1/2,  1]     # Slope
    ])
    
    print("Pairwise Comparison Matrix:")
    print(tabulate(matrix, headers=criteria, showindex=criteria, tablefmt="grid"))

    # Calculate weights using the geometric mean method
    geometric_mean = np.prod(matrix, axis=1) ** (1/n)
    weights = geometric_mean / np.sum(geometric_mean)
    
    # Convert weights to percentages
    weights_percent = weights * 100

    # Create a dictionary of criteria and their corresponding weights
    weights_dict = dict(zip(criteria, weights_percent))

    return weights_dict, matrix

def calculate_consistency_ratio(matrix, weights):
    """
    Calculate the Consistency Ratio (CR) to assess the consistency of the pairwise comparisons.
    """
    n = matrix.shape[0]
    # Calculate lambda_max
    weighted_sum = np.dot(matrix, weights)
    lambda_max = np.sum(weighted_sum / weights) / n

    # Consistency Index (CI)
    consistency_index = (lambda_max - n) / (n - 1)

    # Random Index (RI) values for different n
    random_index = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 
                   6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    ri = random_index.get(n, 1.49)

    # Consistency Ratio (CR)
    consistency_ratio = consistency_index / ri

    return consistency_ratio, consistency_index, lambda_max

def main():
    weights, matrix = calculate_weights()
    
    print("\nCalculated Weights (%):")
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    print(tabulate([[criterion, f"{weight:.2f}%"] for criterion, weight in sorted_weights],
                   headers=["Criterion", "Weight"],
                   tablefmt="grid"))

    # Normalize weights for consistency ratio calculation
    normalized_weights = np.array([weight / 100 for weight in weights.values()])

    # Calculate and print the Consistency Ratio
    cr, ci, lambda_max = calculate_consistency_ratio(matrix, normalized_weights)
    print(f"\nConsistency Measures:")
    print(f"Lambda max: {lambda_max:.4f}")
    print(f"Consistency Index (CI): {ci:.4f}")
    print(f"Consistency Ratio (CR): {cr:.4f}")
    
    if cr < 0.1:
        print("The pairwise comparison matrix is consistent (CR < 0.1).")
    else:
        print("Warning: The pairwise comparison matrix is not consistent (CR >= 0.1).")
        print("Please revise the comparison matrix to improve consistency.")

if __name__ == "__main__":
    main()