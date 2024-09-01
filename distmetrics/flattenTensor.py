import torch

def flattenTensor(source, flattenOption):
    """
    Flattens a tensor according to the specified flattening situation.
    
    Args:
        source (torch.Tensor): Tensor to be flattened.
        flattenOption (dict): Option for different converting situations.
        
    Returns:
        torch.Tensor: Flattened tensor according to the specified situation.
    """
    situation = flattenOption.get('Situation', None)
    
    if situation is None:
        raise ValueError("flattenOption must contain a 'Situation' key.")
    
    if situation == 1:
        # If the shape of a tensor source is [number_of_samples, vectorsize]
        # The shape of the output tensor should be [number_of_samples * vectorsize]
        if source.dim() != 2:
            raise ValueError("Expected tensor of shape [number_of_samples, vectorsize] for Situation 1.")
        return source.view(-1)
    
    elif situation == 2:
        # If the shape of a tensor source is [number_of_samples, rows, columns]
        # The shape of the output tensor should be [number_of_samples * rows * columns]
        if source.dim() != 3:
            raise ValueError("Expected tensor of shape [number_of_samples, rows, columns] for Situation 2.")
        return source.view(-1)
    
    elif situation == 3:
        # If the shape of a tensor source is [number_of_samples, rows, columns]
        # The shape of the output tensor should be [number_of_samples * rows, columns]
        if source.dim() != 3:
            raise ValueError("Expected tensor of shape [number_of_samples, rows, columns] for Situation 3.")
        return source.view(source.size(0) * source.size(1), source.size(2))
    
    elif situation == 4:
        # If the shape of a tensor source is [number_of_samples, rows, columns]
        # The shape of the output tensor should be [number_of_samples * columns, rows]
        if source.dim() != 3:
            raise ValueError("Expected tensor of shape [number_of_samples, rows, columns] for Situation 4.")
        return source.transpose(1, 2).contiguous().view(source.size(0) * source.size(2), source.size(1))
    
    else:
        raise ValueError(f"Unsupported situation: {situation}")

# Example usage
source_tensor_1 = torch.randn(10, 5)
source_tensor_2 = torch.randn(10, 4, 5)

flatten_option_1 = {'Situation': 1}
flatten_option_2 = {'Situation': 2}
flatten_option_3 = {'Situation': 3}
flatten_option_4 = {'Situation': 4}

flattened_tensor_1 = flattenTensor(source_tensor_1, flatten_option_1)
flattened_tensor_2 = flattenTensor(source_tensor_2, flatten_option_2)
flattened_tensor_3 = flattenTensor(source_tensor_2, flatten_option_3)
flattened_tensor_4 = flattenTensor(source_tensor_2, flatten_option_4)

print(flattened_tensor_1.shape)  # Expected: [50]
print(flattened_tensor_2.shape)  # Expected: [200]
print(flattened_tensor_3.shape)  # Expected: [40, 5]
print(flattened_tensor_4.shape)  # Expected: [50, 4]
