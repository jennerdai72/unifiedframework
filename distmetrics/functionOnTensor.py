import torch
#-----------------------------------------------------------------------------------------------------------------------
# Functions on Probability Tensor
#-----------------------------------------------------------------------------------------------------------------------
def TVD_and_Laplacian_Smoothness(input_tensor):
   """
    Applies p_(i) - p_(i-1) to get a new tensor.
    
    Args:
        input_tensor (torch.Tensor): Input tensor of shape [number_of_bins],
     
    Returns:
        torch.Tensor: Input tensor of shape [number_of_bins - 1] .
    """
   return_tensor = input_tensor[1:] - input_tensor[:-1]
   return return_tensor

#-----------------------------------------------------------------------------------------------------------------------
# Functions on Sample Tensor
#-----------------------------------------------------------------------------------------------------------------------
# sample function applies on tensor
def transform_tensor(input_tensor, transform_function, transform_function_option=None):
    """
    Applies a transform function to each item in the input tensor to form a new tensor.
    
    Args:
        input_tensor (torch.Tensor): Input tensor of shape [number_of_samples],
                                     [number_of_samples, vectorsize], or [number_of_samples, rows, columns].
        transform_function (function): Function to apply to each item in the input tensor.
        transform_function_option (dict, optional): Dictionary of options to pass to the transform function.
    
    Returns:
        torch.Tensor: Output tensor with transformed items.
    """
    input_shape = input_tensor.shape
    output_items = []

    if transform_function_option is None:
        transform_function_option = {}

    if input_tensor.dim() == 1:
        # Case where each item is a number
        for i in range(input_shape[0]):
            output_items.append(apply_elementwise_tensor(input_tensor[i],transform_function, transform_function_option))
    elif input_tensor.dim() == 2:
        # Case where each item is a vector
        for i in range(input_shape[0]):
            output_items.append(apply_elementwise_tensor(input_tensor[i, :],transform_function, transform_function_option))
    elif input_tensor.dim() == 3:
        # Case where each item is a matrix
        for i in range(input_shape[0]):
            output_items.append(apply_elementwise_tensor(input_tensor[i, :, :], transform_function,transform_function_option))
    else:
        raise ValueError("Unsupported tensor shape. Expected 1D, 2D, or 3D tensor.")
    
    # Convert the list of output items back to a tensor
    output_tensor = torch.stack(output_items)
    print(output_tensor)
    return output_tensor


def apply_elementwise_tensor(tensor, func, funcOption=None):
    """
    Apply a function element-wise on a tensor to form a new tensor.
    
    Args:
        tensor (torch.Tensor): The input tensor.
        func (function): The function to apply element-wise.
    
    Returns:
        torch.Tensor: A new tensor with the result of the element-wise function application.
    """
    # Create an empty tensor with the same shape to store results
    result = torch.empty_like(tensor)

    # Apply the function element-wise
    if tensor.dim() == 0:
        result = torch.tensor(func(tensor, funcOption))

    elif tensor.dim() == 1:
        for i in range(tensor.shape[0]):
            result[i] = func(tensor[i].item(), funcOption)
    
    elif tensor.dim() == 2:
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                result[i, j] = func(tensor[i, j].item(), funcOption)
    
    elif tensor.dim() == 3:
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                for k in range(tensor.shape[2]):
                    result[i, j, k] = func(tensor[i, j, k].item(), funcOption)
    
    else:
        raise ValueError("Unsupported tensor shape. This function supports tensors with 0,1, 2, or 3 dimensions.")
    
    return result

# sample function applies on 2 tensors
def transform_2tensors(input_tensor1,input_tensor2, transform_function, transform_function_option=None):
    """
    Applies a transform function to each item in the 2 input tensor to form a new tensor.
    
    Args:
        input_tensor1 (torch.Tensor): Input tensor of shape [number_of_samples],
                                     [number_of_samples, vectorsize], or [number_of_samples, rows, columns].
        input_tensor2 (torch.Tensor): Input tensor of shape [number_of_samples],
                                     [number_of_samples, vectorsize], or [number_of_samples, rows, columns].
        transform_function (function): Function to apply to each item in the input tensor.
        transform_function_option (dict, optional): Dictionary of options to pass to the transform function.
    
    Returns:
        torch.Tensor: Output tensor with transformed items.
    """
    input_shape1 = input_tensor1.shape
    input_shape2 = input_tensor2.shape
    output_items = []

    if transform_function_option is None:
        transform_function_option = {}
    
    if input_tensor1.dim() == 1 and input_tensor2.dim() == 1:
        # Case where each item is a number
        for i in range(input_shape1[0]):
            output_items.append(apply_elementwise_2tensors(input_tensor1[i],input_tensor2[i], transform_function,transform_function_option))
    elif input_tensor1.dim() == 2 and input_tensor2.dim() == 2:
        # Case where each item is a vector
        for i in range(input_shape1[0]):
            output_items.append(apply_elementwise_2tensors(input_tensor1[i, :],input_tensor2[i, :], transform_function,transform_function_option))
    elif input_tensor1.dim() == 3 and input_tensor2.dim() == 3:
        # Case where each item is a matrix
        for i in range(input_shape1[0]):
            output_items.append(apply_elementwise_2tensors(input_tensor1[i, :, :],input_tensor2[i, :, :], transform_function,transform_function_option))
    else:
        raise ValueError("Unsupported tensor shape. Expected 1D, 2D, or 3D tensor or unmatched tensor shape of the two input tensors.")
    
    # Convert the list of output items back to a tensor
    output_tensor = torch.stack(output_items)
    print(output_tensor)
    return output_tensor

def apply_elementwise_2tensors(tensor1, tensor2, func, funcOption=None):
    """
    Apply a function element-wise on two tensors to form a new tensor.
    
    Args:
        tensor1 (torch.Tensor): The first input tensor.
        tensor2 (torch.Tensor): The second input tensor.
        func (function): The function to apply element-wise.
    
    Returns:
        torch.Tensor: A new tensor with the result of the element-wise function application.
    """
    # Ensure the tensors are of the same shape
    if tensor1.shape != tensor2.shape:
        raise ValueError("The input tensors must have the same shape.")
    
    # Apply the function element-wise
    result = torch.empty_like(tensor1)  # Create an empty tensor with the same shape
    if tensor1.dim() == 0 and tensor2.dim() == 0:
        result = torch.tensor(func(tensor1.item(), tensor2.item(), funcOption))

    elif tensor1.dim() == 1 and tensor2.dim() == 1:
        for i in range(tensor1.shape[0]):
            result[i] = func(tensor1[i].item(), tensor2[i].item(), funcOption)
    
    elif tensor1.dim() == 2 and tensor2.dim() == 2:
        for i in range(tensor1.shape[0]):
            for j in range(tensor1.shape[1]):
                result[i, j] = func(tensor1[i, j].item(), tensor2[i, j].item(), funcOption)
    
    elif tensor1.dim() == 3 and tensor2.dim() == 3:
        for i in range(tensor1.shape[0]):
            for j in range(tensor1.shape[1]):
                for k in range(tensor1.shape[2]):
                    result[i, j, k] = func(tensor1[i, j, k].item(), tensor2[i, j, k].item(), funcOption)
    
    else:
        raise ValueError("Unsupported tensor shape.")
    
    return result
# Example transform function that uses options
def example_transform(item, factorOption=None):
    """
    Example transform function that multiplies the input by a factor.
    This function can be modified to perform any desired transformation.
    """
    if factorOption is not None:
        factor = factorOption.get("factor")
    else:
        factor = 1
    return item * factor

# Example usage with transform_function_option provided
input_tensor_1d = torch.tensor([1, 2, 3])
transform_function_option = {'factor': 3}
output_tensor_1d = transform_tensor(input_tensor_1d, example_transform, transform_function_option)
print(output_tensor_1d)

input_tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
transform_function_option = {'factor': 4}
output_tensor_2d = transform_tensor(input_tensor_2d, example_transform, transform_function_option)
print(output_tensor_2d)

input_tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
transform_function_option = {'factor': 5}
output_tensor_3d = transform_tensor(input_tensor_3d, example_transform, transform_function_option)
print(output_tensor_3d)
