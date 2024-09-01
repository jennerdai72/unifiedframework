import torch

def calculateDistanceVectorFromOutputsByRefPD(sourceLayerOutputsPD, refNeuronOutputPD, distance_function,distance_function_option=None):
    """
    Calculates the distance vector between a ref neuron output and outputs of all the neurons in the layer
    across all samples.
    
    Args:
        sourceLayerOutputs (torch.Tensor): Outputs of all the neurons in a layer.
        refNeuronOutput (torch.Tensor): Ref output of a neuron.
        distance_function (function): Function to calculate the distance between two tensors.
        
    Returns:
        torch.Tensor: A distance matrix of size [number_of_neurons, number_of_neurons].
    """
     
    num_neurons = sourceLayerOutputsPD.size(1)

    # Initialize the distance matrix
    distance_vector = torch.zeros(num_neurons)
    
    for i in range(num_neurons):
        if sourceLayerOutputsPD.dim() == 2:
            # Shape is [number_of_samples, number_of_neurons]
            distance_vector[i] = distance_function(sourceLayerOutputsPD[:, i], refNeuronOutputPD,distance_function_option)
        elif sourceLayerOutputsPD.dim() == 3:
            # Shape is [number_of_samples, number_of_neurons, vectorsize]
            distance_vector[i] = distance_function(sourceLayerOutputsPD[:, i, :], refNeuronOutputPD,distance_function_option)
        elif sourceLayerOutputsPD.dim() == 4:
            # Shape is [number_of_samples, number_of_neurons, rows, columns]
            distance_vector[i] = distance_function(sourceLayerOutputsPD[:, i, :, :], refNeuronOutputPD,distance_function_option)
        else:
            raise ValueError("Unsupported tensor shape.")
    
    return distance_vector

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

refOutput = torch.tensor([0.2, 0.1, 0.4])
                                      
option = None

distance_vector_2d = calculateDistanceVectorFromOutputsByRefPD(sourceLayerOutputs_2d, refOutput, example_distance_function,option)
print("Distance Matrix (2D):\n", distance_vector_2d)

sourceLayerOutputs_3d = torch.randn(10, 5, 3)
refOutput = torch.randn(10,3)

distance_vector_3d = calculateDistanceVectorFromOutputsByRefPD(sourceLayerOutputs_3d, refOutput, example_distance_function,option)
print("Distance Matrix (3D):\n", distance_vector_3d)

sourceLayerOutputs_4d = torch.randn(10, 5, 4, 4)
refOutput = torch.randn(10,4,4)

distance_vector_4d = calculateDistanceVectorFromOutputsByRefPD(sourceLayerOutputs_4d, refOutput, example_distance_function,option)
print("Distance Matrix (4D):\n", distance_vector_4d)
