import numpy as np
import matplotlib.pyplot as plt
import torch
import pdb
from vision.featurevis import load_feature_image, preprocess_stored_feature_image

def group_neurons(feature_visualizations):
    """
    [UNDER DEVELOPMENT] Provides user interface for semi-automatic neuron grouping

    Args:
        feature_visualizations (list): A list of feature visualization images.

    Returns:
        dict: A dictionary mapping group numbers to lists of neuron indices.
    """

    #TODO: make sure this works
    #TODO: add docstring
    # Display feature visualizations in a grid
    num_neurons = len(feature_visualizations)
    num_cols = 5
    num_rows = (num_neurons + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    
    for i, visualization in enumerate(feature_visualizations):
        row = i // num_cols
        col = i % num_cols
        axes[row, col].imshow(visualization)
        axes[row, col].set_title(f"Neuron {i}")
        axes[row, col].axis("off")
    
    plt.tight_layout()
    plt.show()
    
    # Manually assign neurons to groups based on user input
    num_groups = int(input("Enter the number of groups: "))
    neuron_groups = {}
    
    for i in range(num_neurons):
        group = int(input(f"Enter the group number for Neuron {i}: "))
        if group not in neuron_groups:
            neuron_groups[group] = []
        neuron_groups[group].append(i)
    
    return neuron_groups

def analyze_circuit(model, layer_name, target_neuron, prev_layer_name, k=5):
    """
    Analyzes the circuit of a target neuron by finding the top-k most influential neurons in the previous layer.

    Args:
        model (torch.nn.Module): The PyTorch model.
        layer_name (str): The name of the layer containing the target neuron.
        target_neuron (int): The index of the target neuron.
        prev_layer_name (str): The name of the previous layer.
        top_k (int, optional): The number of top influential neurons to visualize. Defaults to 5.

    Visualizes the feature visualizations and weights of the top-k most influential neurons in the previous layer.
    """

    # Get the weights connecting the previous layer to the target neuron
    target_layer = model.get_layer(layer_name)
    prev_layer = model.get_layer(prev_layer_name)
    weights = target_layer.get_weights()[0][:, :, :, target_neuron]
    
    # Find the top-k most influential neurons in the previous layer
    influential_neurons = find_influential_adjacent_neurons(weights, k)
    
    # Visualize the influential neurons' feature visualizations and weights
    fig, axes = plt.subplots(1, k, figsize=(12, 4))
    
    for i, (neuron_index, strength) in enumerate(influential_neurons):
        visualization = prev_layer.get_weights()[0][:, :, neuron_index]
        axes[i].imshow(visualization)
        axes[i].set_title(f"Neuron {neuron_index}\nStrength: {strength:.2f}")
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.show()

def find_influential_adjacent_neurons(model, target_layer, target_neuron, prev_layer, k):
    """
    Finds the top-k most influential neurons in the previous layer for a target neuron.

    Args:
        model (torch.nn.Module): The PyTorch model.
        target_layer (str): The name of the target layer.
        target_neuron (int): The index of the target neuron.
        prev_layer (str): The name of the previous layer.
        k (int): The number of top influential neurons to return.

    Returns:
        list: A list of tuples containing the influential neuron indices and their corresponding strengths.
    """

    # Get the weights connecting the previous layer to the target layer
    target_layer_weights = model.get_layer(target_layer).get_weights()[0]
    prev_layer_output_shape = model.get_layer(prev_layer).output_shape[3]
    
    # Reshape the weights to match the previous layer's output shape
    weights = target_layer_weights[:, :, :prev_layer_output_shape, target_neuron]
    
    # Calculate the connection strengths and store them in a list
    connection_strengths = []
    num_neurons = weights.shape[2]
    
    for i in range(num_neurons):
        strength = np.abs(weights[:, :, i]).sum()
        connection_strengths.append((i, strength))
    
    # Sort the connection strengths in descending order
    connection_strengths.sort(key=lambda x: x[1], reverse=True)
    
    # Return the top-k most influential neurons
    return connection_strengths[:k]

def find_strongest_influences(model, layer_a, layer_b, channel_num):
    """
    Finds the strongest influences between two layers for a target channel.

    Args:
        model (torch.nn.Module): The PyTorch model.
        layer_a (str): The name of the first layer.
        layer_b (str): The name of the second layer.
        channel_num (int): The index of the target channel in layer_b.

    Returns:
        list: A list of tuples containing the channel indices and their corresponding influence strengths.
    """

    module_a = model
    for submodule in layer_a.split("."):
        module_a = module_a._modules.get(submodule)
    
    num_channels_a = module_a.out_channels
    
    total_effects = torch.zeros(num_channels_a)
    path_counts = torch.zeros(num_channels_a)
    
    current_layer = None
    path_weights = torch.eye(num_channels_a)
    
    for name, module in model.named_modules():
        if name == layer_a:
            current_layer = module
        elif current_layer is not None:
            if isinstance(module, torch.nn.Conv2d):
                norm = torch.linalg.norm(module.weight)
                w = module.weight / norm
                w = w.view(module.out_channels, module.in_channels, -1)
                w = w.sum(dim=-1)  # Sum over spatial dimensions

                if name.endswith('.downsample.0'):  # Downsample layer
                    path_weights = torch.matmul(w, path_weights[:module.in_channels])
                else:  # Regular convolutional layer
                    path_weights = torch.matmul(w, path_weights)
            elif isinstance(module, torch.nn.BatchNorm2d):
                # Skipping batch normalization layers
                pass
            elif isinstance(module, torch.nn.ReLU):
                # Skipping activation layers
                pass
            
            if name == layer_b:
                total_effects += path_weights[:, channel_num]
                path_counts += (path_weights[:, channel_num] != 0).float()
                current_layer = None
                path_weights = torch.eye(num_channels_a)  # Reset path_weights for the next path
    
    total_effects /= path_counts
    
    channel_indices = torch.arange(num_channels_a)
    strongest_channels = torch.stack((channel_indices, total_effects), dim=1)
    
    strongest_channels = strongest_channels[strongest_channels[:, 1].argsort(descending=True)]
    
    # Convert tensor to list of tuples
    strongest_channels = strongest_channels.cpu().numpy()
    strongest_channels = [(int(idx), float(effect)) for idx, effect in strongest_channels]
    
    return strongest_channels

# --- previous influence.py --- co-activation strategy ---
def find_activated_neurons(model, layer1_name, layer2_name, target_channel, image_dir, aggregation='average', k=5):
    """
    Finds highly activated neurons between two layers based on a target channel's feature visualization.

    Args:
        model (torch.nn.Module): The PyTorch model.
        layer1_name (str): The name of the first layer.
        layer2_name (str): The name of the second layer.
        target_channel (int): The index of the target channel in layer2.
        image_dir (str): The directory containing the feature visualization images.
        aggregation (str, optional): The aggregation method used for neuron selection. Defaults to 'average'.
        k (int, optional): The number of top influential neurons to return. Defaults to 5.

    Returns:
        tuple: A tuple containing the indices of the top influential neurons and their corresponding influence scores.
    """

    # Get the specified layers
    layer1 = get_layer_by_name(model, layer1_name)
    layer2 = get_layer_by_name(model, layer2_name)

    # Load the feature visualization image for the target channel in layer2
    feature_image = load_feature_image(image_dir, layer2_name, target_channel, None, aggregation, None)
    feature_image = preprocess_stored_feature_image(feature_image)

    # Register hooks to store activations
    activations = {}
    def save_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    layer1.register_forward_hook(save_activation(layer1_name))
    layer2.register_forward_hook(save_activation(layer2_name))

    # Forward pass with the feature image
    with torch.no_grad():
        model(feature_image)

    # Get the activations of layer1 and layer2
    layer1_activations = activations[layer1_name]
    layer2_activations = activations[layer2_name]

    # Calculate the target activation based on aggregation method
    if aggregation == 'average':
        target_activation = layer2_activations[0, target_channel].mean().item()
    elif aggregation == 'sum':
        target_activation = layer2_activations[0, target_channel].sum().item()

    # Calculate the influence scores
    influence_scores = calculate_coactivation_scores(layer1_activations, target_activation)

    # Get the top k influential neurons/channels
    top_indices = torch.argsort(influence_scores, descending=True)[:k]

    return top_indices.tolist(), influence_scores[top_indices].tolist()

def get_layer_by_name(model, layer_name):
    """
    Retrieves a layer from a PyTorch model by its name.

    Args:
        model (torch.nn.Module): The PyTorch model.
        layer_name (str): The name of the layer to retrieve.

    Returns:
        torch.nn.Module: The layer module.

    Raises:
        ValueError: If the layer with the specified name is not found in the model.
    """

    for name, module in model.named_modules():
        if name == layer_name:
            return module
    raise ValueError(f"No layer named {layer_name} found in the model.")


def calculate_coactivation_scores(layer_activations, target_activation):
    """
    Calculates influence scores based on layer activations and target activation.

    Args:
        layer_activations (torch.Tensor): The activations of the layer.
        target_activation (float): The target activation value.

    Returns:
        torch.Tensor: The influence scores for each neuron in the layer.
    """

    # Normalize the layer activations
    normalized_activations = layer_activations / layer_activations.sum(dim=(2, 3), keepdim=True)

    # Calculate the influence scores
    influence_scores = (normalized_activations * target_activation).sum(dim=(2, 3)).squeeze()

    return influence_scores

if __name__ == '__main__':
    from vision import featurevis

    model = featurevis.load_torchvision_model('resnet50')
    find_strongest_influences(model, 'layer2.1.conv1', 'layer3.3.conv3', 0)