import torch

def getNeuronsDiversityVector(distanceMatrix):
    """
    Calculates the average distance from the output of calculateDistanceMatrix.
    
    Args:
        distanceMatrix (torch.Tensor): The distance matrix.
        
    Returns:
        torch.Tensor: A vector containing the average distances for each neuron.
    """
    # Ensure the distance matrix is a 2D tensor
    if distanceMatrix.dim() != 2:
        raise ValueError("distanceMatrix must be a 2D tensor.")
    
    # Calculate the average distance for each neuron
    average_distances = torch.mean(distanceMatrix, dim=1)
    
    return average_distances

# Example usage
distanceMatrix = torch.tensor([[0.02, 0.01, 0.01, 0.02],
                               [0.10, 0.01, 0.01, 0.10],
                               [0.18, 0.09, 0.01, 0.18],
                               [0.26, 0.17, 0.09, 0.26]])

average_distances = getNeuronsDiversityVector(distanceMatrix)
print("Average Distances:", average_distances)
