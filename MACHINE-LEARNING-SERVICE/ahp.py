import numpy as np
from tabulate import tabulate

def calculate_weights():
    # Define the criteria based on the provided datasets
    criteria = ['Soils', 'Geology', 'Digital Elevation Model', 'Protected Areas', 'Rivers', 'Roads', 'Land Use/Land Cover']
    n = len(criteria)
    
    #comparison matrix
    matrix = np.array([
        [1,    1,    3,    5,    7,    5,    1  ],  # Soils
        [1,    1,    3,    5,    7,    5,    1  ],  # Geology
        [1/3,  1/3,  1,    3,    5,    3,    1/3],  # Digital Elevation Model
        [1/5,  1/5,  1/3,  1,    3,    1,    1/5],  # Protected Areas
        [1/7,  1/7,  1/5,  1/3,  1,    1/3,  1/7],  # Rivers
        [1/5,  1/5,  1/3,  1,    3,    1,    1/5],  # Roads
        [1,    1,    3,    5,    7,    5,    1  ]   # Land Use/Land Cover
    ])

    print("Comparison Matrix:")
    print(tabulate(matrix, headers=criteria, showindex=criteria, tablefmt="grid"))

    # Calculate weights using the eigenvector method
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_index = np.argmax(eigenvalues.real)
    weights = eigenvectors[:, max_index].real
    weights = weights / np.sum(weights)

    # Convert weights to percentages
    weights_percent = weights * 100

    # Create a dictionary of criteria and their corresponding weights
    weights_dict = dict(zip(criteria, weights_percent))

    return weights_dict, matrix

def calculate_consistency_ratio(matrix, weights):
    n = matrix.shape[0]
    # Calculate lambda_max
    weighted_sum = np.dot(matrix, weights)
    lambda_max = np.sum(weighted_sum / weights) / n

    # Consistency Index (CI)
    consistency_index = (lambda_max - n) / (n - 1)

    # Random Index (RI) values for different n
    random_index = {3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    ri = random_index.get(n, 1.49)  # Default to 1.49 if n > 10

    # Consistency Ratio (CR)
    consistency_ratio = consistency_index / ri

    return consistency_ratio

def main():
    weights, matrix = calculate_weights()
    print("\nWeights (%):")
    print(tabulate([[criterion, f"{weight:.2f}%"] for criterion, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True)], headers=["Criterion", "Weight"], tablefmt="grid"))

    # Normalize weights for consistency ratio calculation
    normalized_weights = np.array([weight / 100 for weight in weights.values()])

    # Calculate and print the Consistency Ratio
    consistency_ratio = calculate_consistency_ratio(matrix, normalized_weights)
    print(f"\nConsistency Ratio: {consistency_ratio:.4f}")
    if consistency_ratio < 0.1:
        print("The comparison matrix is consistent (CR < 0.1)")
    else:
        print("Warning: The comparison matrix is not consistent (CR >= 0.1)")
        print("Please revise the comparison matrix to improve consistency.")

if __name__ == "__main__":
    main()