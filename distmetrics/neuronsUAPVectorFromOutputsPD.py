import torch

def calculateUAPVectorByOutputsPD(sourceLayerOutputsPD, uap_function, uap_function_option=None):
    """
    Calculates the UAP vector for each neuron in the probability distributions of the source sample set.
    
    Args:
        sourceLayerOutputsPD (torch.Tensor): Outputs of the probability distributions of the source sample set.
        uap_function (function): Function to calculate the UAP for a tensor.
        uap_function_option (any): Additional option for the UAP function.
        
    Returns:
        torch.Tensor: A UAP vector of size [number_of_neurons].
    """
    num_neurons = sourceLayerOutputsPD.size(1)

    # Initialize the UAP vector
    uap_vector = torch.zeros(num_neurons)
    
    for i in range(num_neurons):
        if sourceLayerOutputsPD.dim() == 2:
            # Shape is [number_of_bins, number_of_neurons]
            uap_vector[i] = uap_function(sourceLayerOutputsPD[:, i], uap_function_option)
        elif sourceLayerOutputsPD.dim() == 3:
            # Shape is [number_of_bins, number_of_neurons, vectorsize]
            uap_vector[i] = uap_function(sourceLayerOutputsPD[:, i, :], uap_function_option)
        elif sourceLayerOutputsPD.dim() == 4:
            # Shape is [number_of_samples, number_of_neurons, rows, columns]
            uap_vector[i] = uap_function(sourceLayerOutputsPD[:, i, :, :], uap_function_option)
        else:
            raise ValueError("Unsupported tensor shape.")
    
    return uap_vector

# Example UAP function
def example_uap_function(tensor, option=None):
    """
    Example UAP function that calculates the mean of the tensor.
    
    Args:
        tensor (torch.Tensor): Input tensor.
        
    Returns:
        float: Calculated UAP value.
    """
    return torch.mean(tensor).item()

# Example usage
sourceLayerOutputs_2d = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                      [0.5, 0.6, 0.7, 0.8],
                                      [0.9, 1.0, 1.1, 1.2]])

option = None

uap_vector_2d = calculateUAPVectorByOutputsPD(sourceLayerOutputs_2d, example_uap_function, option)
print("UAP Vector (2D):\n", uap_vector_2d)

sourceLayerOutputs_3d = torch.randn(10, 5, 3)
uap_vector_3d = calculateUAPVectorByOutputsPD(sourceLayerOutputs_3d, example_uap_function, option)
print("UAP Vector (3D):\n", uap_vector_3d)

sourceLayerOutputs_4d = torch.randn(10, 5, 4, 4)
uap_vector_4d = calculateUAPVectorByOutputsPD(sourceLayerOutputs_4d, example_uap_function, option)
print("UAP Vector (4D):\n", uap_vector_4d)
