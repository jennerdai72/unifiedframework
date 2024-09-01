import torch

def calculateSimilarityVector(sourceLayerOutputs, targetLayerOutputs, distance_function):
    """
    Calculates the distance between the outputs of a specific layer of a neural network 
    coming across the source sample set and the target sample set, iterating over neurons.
    
    Args:
        sourceLayerOutputs (torch.Tensor): Outputs of the source sample set.
        targetLayerOutputs (torch.Tensor): Outputs of the target sample set.
        distance_function (function): Function to calculate the distance between two tensors.
        
    Returns:
        torch.Tensor: A vector of similarity for each neuron across the source sample set and the target sample set.
    """
    # Ensure the source and target outputs have the same shape
    if sourceLayerOutputs.shape != targetLayerOutputs.shape:
        raise ValueError("Source and target outputs must have the same shape.")
    
    # Determine the shape of the outputs
    if sourceLayerOutputs.dim() == 2:
        # Shape is [number_of_samples, number_of_neurons]
        distances = torch.stack([distance_function(sourceLayerOutputs[:, j], targetLayerOutputs[:, j])
                                 for j in range(sourceLayerOutputs.size(1))])
    elif sourceLayerOutputs.dim() == 3:
        # Shape is [number_of_samples, number_of_neurons, vectorsize]
        distances = torch.stack([distance_function(sourceLayerOutputs[:, j, :], targetLayerOutputs[:, j, :])
                                 for j in range(sourceLayerOutputs.size(1))])
    elif sourceLayerOutputs.dim() == 4:
        # Shape is [number_of_samples, number_of_neurons, rows, columns]
        distances = torch.stack([distance_function(sourceLayerOutputs[:, j, :, :], targetLayerOutputs[:, j, :, :])
                                 for j in range(sourceLayerOutputs.size(1))])
    else:
        raise ValueError("Unsupported tensor shape.")
    
    return distances

# Example distance function
def example_distance_function(tensor1, tensor2):
    """
    Example distance function that calculates the mean squared error between two tensors.
    
    Args:
        tensor1 (torch.Tensor): First tensor.
        tensor2 (torch.Tensor): Second tensor.
        
    Returns:
        torch.Tensor: Calculated distance.
    """
    return torch.mean((tensor1 - tensor2) ** 2)

# Example usage
sourceLayerOutputs_2d = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                      [0.5, 0.6, 0.7, 0.8],
                                      [0.9, 1.0, 1.1, 1.2]])

targetLayerOutputs_2d = torch.tensor([[0.2, 0.1, 0.4, 0.3],
                                      [0.6, 0.5, 0.8, 0.7],
                                      [1.0, 0.9, 1.2, 1.1]])

distances_2d = calculateSimilarityVector(sourceLayerOutputs_2d, targetLayerOutputs_2d, example_distance_function)
print("Distances (2D):", distances_2d)

sourceLayerOutputs_3d = torch.randn(10, 5, 3)
targetLayerOutputs_3d = torch.randn(10, 5, 3)

distances_3d = calculateSimilarityVector(sourceLayerOutputs_3d, targetLayerOutputs_3d, example_distance_function)
print("Distances (3D):", distances_3d)

sourceLayerOutputs_4d = torch.randn(10, 5, 4, 4)
targetLayerOutputs_4d = torch.randn(10, 5, 4, 4)

distances_4d = calculateSimilarityVector(sourceLayerOutputs_4d, targetLayerOutputs_4d, example_distance_function)
print("Distances (4D):", distances_4d)
