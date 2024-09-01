import torch
import numpy as np

def outputs_from_neuron_to_probability(samples, probabilityOption=None):
    """
    Converts a tensor of samples to a tensor of probabilities.
    
    Args:
        samples (torch.Tensor): Input tensor of shape [number_of_samples].
        probabilityOption (dict): Dictionary with options for the conversion.
            - 'Num_bins': Number of bins or items in the output tensor.
            - 'Type': Type of distribution function ('PMF', 'PDF', 'CDF', 'EDF').

    Returns:
        torch.Tensor: Output tensor of probabilities.
    """
    if probabilityOption is None:
        probabilityOption = {}
        
    num_bins = probabilityOption.get('Num_bins', 10)
    dist_type = probabilityOption.get('PType', 'PMF')
    
    if dist_type not in ['PMF', 'PDF', 'CDF', 'EDF']:
        raise ValueError(f"Unknown distribution type: {dist_type}")
    
    samples_np = samples.numpy()
    
    if dist_type == 'PMF':
        # PMF: Probability Mass Function
        counts = np.bincount(samples_np.astype(int), minlength=num_bins)
        probabilities = counts / np.sum(counts)
    
    elif dist_type == 'PDF':
        # PDF: Probability Density Function (histogram)
        counts, bin_edges = np.histogram(samples_np, bins=num_bins, density=True)
        probabilities = counts * np.diff(bin_edges)
    
    elif dist_type == 'CDF':
        # CDF: Cumulative Distribution Function
        counts, bin_edges = np.histogram(samples_np, bins=num_bins, density=True)
        cdf = np.cumsum(counts * np.diff(bin_edges))
        probabilities = cdf / cdf[-1]
    
    elif dist_type == 'EDF':
        # EDF: Empirical Distribution Function
        sorted_samples = np.sort(samples_np)
        probabilities = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
        # Interpolate to fit the number of bins
        probabilities = np.interp(np.linspace(0, 1, num_bins), np.linspace(0, 1, len(probabilities)), probabilities)
    # print(probabilities)
    # print(torch.tensor(probabilities))
    return torch.tensor(probabilities)

# Example usage:
samples = torch.tensor([1, 2, 2, 3, 4, 4, 4, 5, 5, 6])

# PMF Example
probabilityOption = {'num_bins': 7, 'PType': 'PMF'}
print(outputs_from_neuron_to_probability(samples, probabilityOption))

# PDF Example
probabilityOption = {'num_bins': 7, 'PType': 'PDF'}
print(outputs_from_neuron_to_probability(samples, probabilityOption))

# CDF Example
probabilityOption = {'num_bins': 7, 'PType': 'CDF'}
print(outputs_from_neuron_to_probability(samples, probabilityOption))

# EDF Example
probabilityOption = {'num_bins': 7, 'PType': 'EDF'}
print(outputs_from_neuron_to_probability(samples, probabilityOption))
