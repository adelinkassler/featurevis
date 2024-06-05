import numpy as np
import matplotlib.pyplot as plt
import torch
import pdb

def group_neurons(feature_visualizations):
    # Used for manually grouping neurons based on visual similarity
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

def analyze_circuit(model, layer_name, target_neuron, prev_layer_name, top_k=5):
    # Get the weights connecting the previous layer to the target neuron
    target_layer = model.get_layer(layer_name)
    prev_layer = model.get_layer(prev_layer_name)
    weights = target_layer.get_weights()[0][:, :, :, target_neuron]
    
    # Find the top-k most influential neurons in the previous layer
    influential_neurons = find_influential_adjacent_neurons(weights, top_k)
    
    # Visualize the influential neurons' feature visualizations and weights
    fig, axes = plt.subplots(1, top_k, figsize=(12, 4))
    
    for i, (neuron_index, strength) in enumerate(influential_neurons):
        visualization = prev_layer.get_weights()[0][:, :, neuron_index]
        axes[i].imshow(visualization)
        axes[i].set_title(f"Neuron {neuron_index}\nStrength: {strength:.2f}")
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.show()

def find_influential_adjacent_neurons(model, target_layer, target_neuron, prev_layer, top_k):
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
    return connection_strengths[:top_k]

def find_strongest_influences(model, layer_a, layer_b, channel_num):
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

if __name__ == '__main__':
    from vision import featurevis

    model = featurevis.load_torchvision_model('resnet50')
    find_strongest_influences(model, 'layer2.1.conv1', 'layer3.3.conv3', 0)