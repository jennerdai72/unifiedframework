import torch
from .outputsNeuronToProbabilities import outputs_from_neuron_to_probability


def outputs_from_layer_to_probability(outputs, probabilityOption=None):
    """
    Convert the outputs of a layer in a neural network to probability distributions for each neuron.
    
    Args:
        outputs (torch.Tensor): The outputs of a layer with shape [number_of_samples, number_of_neurons].
        probabilityOption (dict or None): Options for probability conversion.
        
    Returns:
        torch.Tensor: The probability distributions of all neurons with shape [number_of_bins, number_of_neurons].
    """
    number_of_samples, number_of_neurons = outputs.shape
    
    # List to hold the probability distributions of each neuron
    neuron_probabilities = []
    
    # Convert the outputs of each neuron to a probability distribution
    for neuron_idx in range(number_of_neurons):
        neuron_outputs = outputs[:, neuron_idx]
        neuron_probability = outputs_from_neuron_to_probability(neuron_outputs, probabilityOption)
        neuron_probabilities.append(neuron_probability)
    
    # Stack the probabilities to form a tensor with shape [number_of_bins, number_of_neurons]
    probabilities_tensor = torch.stack(neuron_probabilities, dim=1)
    
    return probabilities_tensor

def output_from_layer_to_probability(output, probabilityOption=None):
    """
    Convert the output of a neuron in a neural network to probability distributions.
    
    Args:
        output (torch.Tensor): The output of a neuron with shape [number_of_samples].
        probabilityOption (dict or None): Options for probability conversion.
        
    Returns:
        torch.Tensor: The probability distributions of the neuron with shape [number_of_bins].
    """
  
    
    # Convert the outputs of each neuron to a probability distribution
    
    neuron_probability = outputs_from_neuron_to_probability(output, probabilityOption)
    
    
    return neuron_probability

# Example usage
# Simulate some outputs
number_of_samples = 100
number_of_neurons = 10
outputs = torch.randn(number_of_samples, number_of_neurons)

# Convert the layer outputs to probability distributions
probabilityOption = {"PType":"PDF","Num_bins": 50}
probabilities = outputs_from_layer_to_probability(outputs, probabilityOption)
print(probabilities.shape)  # Should be [50, 10]
