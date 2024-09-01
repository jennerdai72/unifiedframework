import torch

def neuronsSimilarityFilter(similarityVector, threshold, n, filterOption):
    """
    Filters neurons based on their similarity measures.
    
    Args:
        similarityVector (torch.Tensor): A vector where each item is the similarity measure of a neuron.
        threshold (float): The value used to select neurons.
        n (int): Number of top or lowest neurons to select.
        filterOption (dict): Option to determine the filtering criteria.
        
    Returns:
        list: Indices of the selected neurons.
    """
    if not isinstance(similarityVector, torch.Tensor):
        raise ValueError("similarityVector must be a torch.Tensor.")
    
    option = filterOption.get('Threshold')
    selected_indices = []

    if option == '>=':
        selected_indices = (similarityVector >= threshold).nonzero(as_tuple=True)[0].tolist()
    elif option == '<=':
        selected_indices = (similarityVector <= threshold).nonzero(as_tuple=True)[0].tolist()
    elif option == 'Top':
        _, indices = torch.topk(similarityVector, n, largest=True)
        selected_indices = indices.tolist()
    elif option == 'Lower':
        _, indices = torch.topk(similarityVector, n, largest=False)
        selected_indices = indices.tolist()
    else:
        raise ValueError("Invalid filterOption value. Must be one of '>=', '<=', 'Top', 'Lower'.")
    
    return selected_indices

# Example usage
similarityVector = torch.tensor([0.0150, 0.0550, 0.1150, 0.1950])
threshold = 0.05
n = 2

# Using threshold >=
filterOption = {'Threshold': '>='}
selected_indices_gte = neuronsSimilarityFilter(similarityVector, threshold, n, filterOption)
print("Selected Indices (>= threshold):", selected_indices_gte)

# Using threshold <=
filterOption = {'Threshold': '<='}
selected_indices_lte = neuronsSimilarityFilter(similarityVector, threshold, n, filterOption)
print("Selected Indices (<= threshold):", selected_indices_lte)

# Using top n
filterOption = {'Threshold': 'Top'}
selected_indices_top = neuronsSimilarityFilter(similarityVector, threshold, n, filterOption)
print("Selected Indices (Top n):", selected_indices_top)

# Using lowest n
filterOption = {'Threshold': 'Lower'}
selected_indices_lower = neuronsSimilarityFilter(similarityVector, threshold, n, filterOption)
print("Selected Indices (Lowest n):", selected_indices_lower)
