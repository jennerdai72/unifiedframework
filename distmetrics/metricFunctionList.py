import torch

def p_norm_metric(tensor, metricOption=None):
    """
    Computes the p-norm of each item in the input tensor.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape [], [vectorsize], or [rows, columns].
        p (float): The p-value for the p-norm.
    
    Returns:
        torch.Tensor: Output tensor of shape [number_of_samples], where each item is the p-norm of the corresponding input item.
    """
    if metricOption is not None:
        p = metricOption.get("p")
    else:
        p = 2
    if tensor.dim() == 0:
        # Case where each item is a number
        result = tensor.abs().pow(p).pow(1 / p)
    elif tensor.dim() == 1:
        # Case where each item is a vector
        result = tensor.abs().pow(p).sum(dim=1).pow(1 / p)
    elif tensor.dim() == 2:
        # Case where each item is a matrix
        result = tensor.abs().pow(p).sum(dim=(1, 2)).pow(1 / p)
    else:
        raise ValueError("Unsupported tensor shape. Expected 0D, 1D, or 2D tensor.")
    
    return result