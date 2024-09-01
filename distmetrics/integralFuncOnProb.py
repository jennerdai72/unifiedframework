import torch

def integral_funcOnProb_with_options(functions_list, function_options_list, source_probability_tensor, target_probability_tensor=None):
    """
    Applies functions from a list of lists to items from both source and optionally target probability tensors using the provided options,
    sums the results, and stores them in an output list of lists.
    
    Args:
        functions_list (list of lists): A list of lists where each element is a function.
        function_options_list (list of lists): A list of lists where each element is a dictionary of options for the corresponding function.
        source_probability_tensor (torch.Tensor): Source probability tensor to apply the functions to.
        target_probability_tensor (torch.Tensor, optional): Target probability tensor to apply the functions to. If None, only the source tensor is used.
    
    Returns:
        list of lists: An output list of lists with the summed results of applying the functions.
    """
    output = []

    for function_list, options_list in zip(functions_list, function_options_list):
        sub_output = []
        for func, options in zip(function_list, options_list):
            if target_probability_tensor is not None:
                result = sum(func(source_item, target_item, options) 
                             for source_item, target_item in zip(source_probability_tensor, target_probability_tensor))
            else:
                result = sum(func(source_item, options) for source_item in source_probability_tensor)
                
            sub_output.append(result)
        output.append(sub_output)
    
    return output

# Example functions
def example_function(source_item, target_item=None, Option=None):
    """
    Example function that multiplies the source item by a factor and optionally adds the target item.
    """
    if Option is not None:
        factor = Option.get("factor")
    else:
        factor = 1
    return source_item * factor #+ (target_item if target_item is not None else 0)

def another_example_function(source_item, target_item=None, Option=None):
    """
    Example function that adds a given value to the source item and optionally subtracts the target item.
    """
    if Option is not None:
        addend = Option.get("addend")
    else:
        addend = 0
    return source_item + addend #- (target_item if target_item is not None else 0)

# Example usage
functions_list = [
    [example_function, another_example_function],
    [another_example_function, example_function]
]

function_options_list = [
    [{'factor': 2}, {'addend': 3}],
    [{'addend': 1}, {'factor': 4}]
]

source_probability_tensor = torch.tensor([0.1, 0.2, 0.3, 0.4])
target_probability_tensor = torch.tensor([0.25, 0.25, 0.25, 0.25])

# Case with both source and target tensors
output_with_target = integral_funcOnProb_with_options(functions_list, function_options_list, source_probability_tensor, target_probability_tensor)
print("Output with target tensor:", output_with_target)

# Case with only the source tensor
output_without_target = integral_funcOnProb_with_options(functions_list, function_options_list, source_probability_tensor)
print("Output without target tensor:", output_without_target)
