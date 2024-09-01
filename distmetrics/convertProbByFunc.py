import torch
import numpy as np
import matplotlib.pyplot as plt

from .outputsNeuronToProbabilities import outputs_from_neuron_to_probability


# This is for Appendix B
def convert_to_probabilities(tensor,option=None):
    """
    Convert a tensor to a probability distribution using sample_to_probability.
    
    Args:
        tensor (torch.Tensor): Input tensor.
        
    Returns:
        torch.Tensor: Tensor representing the probability distribution.
    """
    if option is not None:
        option = {}
    probabilityOption = {'num_bins': option.get("num_bins",100), 'PType': option.get("PType","PDF")}
    return outputs_from_neuron_to_probability(tensor,probabilityOption)

def convertProbByFunc(source, conversionFunction, conversionFunctionOption, metric, metricOption, pOtion=None):
    """
    Converts a source tensor using a conversion function to get a target tensor.
    Applies a metric function to evaluate the source and target tensors separately.
    Converts the results of metric function to probability distributions and saves the plots.
    
    Args:
        source (torch.Tensor): Source tensor, output of a neuron across all samples.
        conversionFunction (function): Function to convert the source tensor to a target tensor.
        conversionFunctionOption (dict): Options for the conversion function.
        metric (function): Metric function to evaluate source and target tensors.
        metricOption (dict): Options for the metric function.
    """
    # Convert the source tensor to target tensor
    target = conversionFunction(source, conversionFunctionOption)
    
    # Apply metric function separately
    source_metric_result = metric(source, metricOption)
    target_metric_result = metric(target, metricOption)
    # print(f"Source Metric Result: {source_metric_result}")
    # print(f"Target Metric Result: {target_metric_result}")
    
    # Convert metric results to probability distributions
    source_prob = convert_to_probabilities(source_metric_result,pOtion)
    target_prob = convert_to_probabilities(target_metric_result,pOtion)
    
    # Get the name of the conversion function
    conversion_function_name = conversionFunction.__name__
    
    # Plot the probability distributions
    samplesSource = np.arange(source_prob.shape[0])
    samplesTarget = np.arange(target_prob.shape[0])
     
    plt.figure(figsize=(10, 5))
    plt.bar(samplesSource, source_prob.numpy(), alpha=0.7, label='Source Probability')
    plt.xlabel('Samples')
    plt.ylabel('Probability')
    plt.title('Source Probability Distribution')
    plt.legend()
    plt.savefig(f"source_probability_distribution_{conversion_function_name}.png")
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.bar(samplesTarget, target_prob.numpy(), alpha=0.7, label='Target Probability')
    plt.xlabel('Samples')
    plt.ylabel('Probability')
    plt.title('Target Probability Distribution')
    plt.legend()
    plt.savefig(f"target_probability_distribution_{conversion_function_name}.png")
    plt.close()

# Example conversion function
def example_conversion_function(source, Option=None):
    """
    Example conversion function that multiplies the source tensor by a factor.
    
    Args:
        source (torch.Tensor): Source tensor.
        factor (float): Multiplication factor.
        
    Returns:
        torch.Tensor: Converted tensor.
    """
    if Option is not None:
        factor = Option.get("factor")
    else:
        factor = 1
    return source * factor

# Example metric function
def example_metric_function(tensor, Option=None):
    """
    Example metric function that calculates the mean squared error with respect to a tensor of ones.
    
    Args:
        tensor (torch.Tensor): Input tensor.
        factor (float): Factor to scale the result.
        
    Returns:
        torch.Tensor: Calculated metric tensor.
    """
    if Option is not None:
        factor = Option.get("factor")
    else:
        factor = 1
    reference_tensor = torch.ones_like(tensor)
    return factor * (tensor - reference_tensor) ** 2

# Example usage
source_tensor = torch.randn(100)

conversion_function_option = {'factor': 2}
metric_option = {'factor': 1}

convertProbByFunc(source_tensor, example_conversion_function, conversion_function_option, example_metric_function, metric_option)
