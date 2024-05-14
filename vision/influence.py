import torch
import torch.nn as nn
from .featurevis import load_feature_image, preprocess_stored_feature_image

def find_influential_neurons(model, layer1_name, layer2_name, target_channel, image_dir, aggregation='average', k=5):
    # Get the specified layers
    layer1 = get_layer_by_name(model, layer1_name)
    layer2 = get_layer_by_name(model, layer2_name)

    # Load the feature visualization image for the target channel in layer2
    feature_image = load_feature_image(image_dir, layer2_name, target_channel, None, aggregation)
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
    influence_scores = calculate_influence_scores(layer1_activations, target_activation)

    # Get the top k influential neurons/channels
    top_indices = torch.argsort(influence_scores, descending=True)[:k]

    return top_indices.tolist(), influence_scores[top_indices].tolist()

def get_layer_by_name(model, layer_name):
    for name, module in model.named_modules():
        if name == layer_name:
            return module
    raise ValueError(f"No layer named {layer_name} found in the model.")


def calculate_influence_scores(layer_activations, target_activation):
    # Normalize the layer activations
    normalized_activations = layer_activations / layer_activations.sum(dim=(2, 3), keepdim=True)

    # Calculate the influence scores
    influence_scores = (normalized_activations * target_activation).sum(dim=(2, 3)).squeeze()

    return influence_scores