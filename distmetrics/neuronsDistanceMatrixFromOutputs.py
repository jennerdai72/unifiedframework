import torch

def calculateDistanceMatrixByOutputs(sourceLayerOutputs, targetLayerOutputs, distance_function,distance_function_option=None):
    """
    Calculates the distance matrix between each neuron and all the neurons in the layer
    for the source and target sample sets.
    
    Args:
        sourceLayerOutputs (torch.Tensor): Outputs of the source sample set.
        targetLayerOutputs (torch.Tensor): Outputs of the target sample set.
        distance_function (function): Function to calculate the distance between two tensors.
        
    Returns:
        torch.Tensor: A distance matrix of size [number_of_neurons, number_of_neurons].
    """
    # Ensure the source and target outputs have the same shape
    if sourceLayerOutputs.shape != targetLayerOutputs.shape:
        raise ValueError("Source and target outputs must have the same shape.")
    
    num_neurons = sourceLayerOutputs.size(1)

    # Initialize the distance matrix
    distance_matrix = torch.zeros(num_neurons, num_neurons)
    
    for i in range(num_neurons):
        for j in range(num_neurons):
            if sourceLayerOutputs.dim() == 2:
                # Shape is [number_of_samples, number_of_neurons]
                distance_matrix[i, j] = distance_function(sourceLayerOutputs[:, i], targetLayerOutputs[:, j],distance_function_option)
            elif sourceLayerOutputs.dim() == 3:
                # Shape is [number_of_samples, number_of_neurons, vectorsize]
                distance_matrix[i, j] = distance_function(sourceLayerOutputs[:, i, :], targetLayerOutputs[:, j, :],distance_function_option)
            elif sourceLayerOutputs.dim() == 4:
                # Shape is [number_of_samples, number_of_neurons, rows, columns]
                distance_matrix[i, j] = distance_function(sourceLayerOutputs[:, i, :, :], targetLayerOutputs[:, j, :, :],distance_function_option)
            else:
                raise ValueError("Unsupported tensor shape.")
    
    return distance_matrix

# Example distance function
def example_distance_function(tensor1, tensor2,option=None):
    """
    Example distance function that calculates the mean squared error between two tensors.
    
    Args:
        tensor1 (torch.Tensor): First tensor.
        tensor2 (torch.Tensor): Second tensor.
        
    Returns:
        float: Calculated distance.
    """
    return torch.mean((tensor1 - tensor2) ** 2).item()

# Example usage
sourceLayerOutputs_2d = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                      [0.5, 0.6, 0.7, 0.8],
                                      [0.9, 1.0, 1.1, 1.2]])

targetLayerOutputs_2d = torch.tensor([[0.2, 0.1, 0.4, 0.3],
                                      [0.6, 0.5, 0.8, 0.7],
                                      [1.0, 0.9, 1.2, 1.1]])
option = None

distance_matrix_2d = calculateDistanceMatrixByOutputs(sourceLayerOutputs_2d, targetLayerOutputs_2d, example_distance_function,option)
print("Distance Matrix (2D):\n", distance_matrix_2d)

sourceLayerOutputs_3d = torch.randn(10, 5, 3)
targetLayerOutputs_3d = torch.randn(10, 5, 3)

distance_matrix_3d = calculateDistanceMatrixByOutputs(sourceLayerOutputs_3d, targetLayerOutputs_3d, example_distance_function,option)
print("Distance Matrix (3D):\n", distance_matrix_3d)

sourceLayerOutputs_4d = torch.randn(10, 5, 4, 4)
targetLayerOutputs_4d = torch.randn(10, 5, 4, 4)

distance_matrix_4d = calculateDistanceMatrixByOutputs(sourceLayerOutputs_4d, targetLayerOutputs_4d, example_distance_function,option)
print("Distance Matrix (4D):\n", distance_matrix_4d)
