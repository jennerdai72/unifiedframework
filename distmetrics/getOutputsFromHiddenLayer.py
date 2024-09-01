import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def get_hidden_layer_outputs(model, data_loader,dataitem_conversion_function, model_class=None, layer=None, layer_index=None, model_dims=None):
    """
    Get the outputs of a hidden layer across all samples.

    Args:
        model (nn.Module or str): The neural network model or file name to load the model from.
        data_loader (DataLoader): DataLoader for the dataset.
        model_class (nn.Module): The class of the model to instantiate if loading from a file.
        layer (str): The name of the hidden layer to capture outputs from.
        layer_index (int): The index of the hidden layer to capture outputs from.
        model_dims (dict): Dictionary containing model dimensions (input_size, hidden_size, output_size).

    Returns:
        torch.Tensor: The outputs of the hidden layer for all samples.
    """
    if isinstance(model, str):
        if model_class is None or model_dims is None:
            raise ValueError("model_class and model_dims must be provided when loading model from a file")
        # Load the model from the file
        model_instance = model_class(**model_dims)
        model_instance.load_state_dict(torch.load(model, map_location=torch.device('cpu')))
        model = model_instance

    outputs = []
    
    def hook_fn(module, input, output):
        outputs.append(output)

    # Register hook to the specified hidden layer
    if layer is not None:
        hook = getattr(model, layer).register_forward_hook(hook_fn)
    elif layer_index is not None:
        hook = list(model.children())[layer_index].register_forward_hook(hook_fn)
    else:
        raise ValueError("Either layer or layer_index must be provided")

    # Forward pass through the model to collect outputs
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            inputs, _ = data
            inputs = dataitem_conversion_function(inputs)
            model(inputs)
    
    # Remove hook
    hook.remove()
    
    # stack all outputs
    hidden_outputs = torch.cat(outputs, dim=0)
    return hidden_outputs

#------------------------------------------------------------
# data item convert function list
#------------------------------------------------------------

def flattenInputs(inputs):
    return inputs.reshape(-1,28*28)

#=========================End=================================

class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)  # Assuming ReLU activation
        x = self.fc2(x)
        return x
# # Example usage
# input_size = 10
# hidden_size = 128
# output_size = 2
# batch_size = 32

# model_dims = {
#     'input_size': input_size,
#     'hidden_size': hidden_size,
#     'output_size': output_size
# }

# # Create some example data
# inputs = torch.rand(100, input_size)
# targets = torch.randint(0, 2, (100,))
# dataset = TensorDataset(inputs, targets)
# data_loader = DataLoader(dataset, batch_size=batch_size)

# # Save an example model to a file for demonstration
# example_model = SimpleModel(**model_dims)
# torch.save(example_model.state_dict(), f'model_hidden_size_{hidden_size}.pth')

# # Get the hidden layer outputs from a model instance by layer name
# hidden_layer_outputs_instance = get_hidden_layer_outputs(example_model, data_loader, layer='fc1')
# print(hidden_layer_outputs_instance.shape)  # Should be (100, hidden_size)

# # Get the hidden layer outputs from a model instance by layer index
# hidden_layer_outputs_instance_index = get_hidden_layer_outputs(example_model, data_loader, layer_index=0)
# print(hidden_layer_outputs_instance_index.shape)  # Should be (100, hidden_size)

# # Get the hidden layer outputs from a model file by layer name
# hidden_layer_outputs_file = get_hidden_layer_outputs(f'model_hidden_size_{hidden_size}.pth', data_loader, model_class=SimpleModel, layer='fc1', model_dims=model_dims)
# print(hidden_layer_outputs_file.shape)  # Should be (100, hidden_size)

# # Get the hidden layer outputs from a model file by layer index
# hidden_layer_outputs_file_index = get_hidden_layer_outputs(f'model_hidden_size_{hidden_size}.pth', data_loader, model_class=SimpleModel, layer_index=0, model_dims=model_dims)
# print(hidden_layer_outputs_file_index.shape)  # Should be (100, hidden_size)
