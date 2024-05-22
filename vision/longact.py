import os
import torch
from typing import List, Dict
import matplotlib.pyplot as plt
from vision.featurevis import load_torchvision_model
from vision import utils

def get_longitudinal_activations(model_name: str, checkpoint_paths: List[str], layer_name: str, channel_num: int, 
                             feature_image: torch.Tensor, device: str = 'cpu') -> Dict[str, float]:
    activations = {}
    
    for checkpoint_path in checkpoint_paths:
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        model = load_torchvision_model(model_name=model_name, checkpoint_path=checkpoint_path, device=device)
        model.eval()
        
        # Find the target layer
        target_layer = model
        for submodule in layer_name.split('.'):
            target_layer = target_layer._modules.get(submodule)
        
        activation = None
        
        def hook(module, input, output):
            nonlocal activation
            activation = output[:, channel_num].mean().item()
        
        handle = target_layer.register_forward_hook(hook)
        
        feature_image = feature_image.to(device)
        with torch.no_grad():
            model(feature_image)
        
        handle.remove()
        
        activations[checkpoint_name] = activation
    
    return activations

def plot_longitudinal_activations(activations: Dict[str, float], layer_name: str, channel_num: int, 
                                  output_path: str = None) -> None:
    checkpoint_names = list(activations.keys())
    checkpoint_names.sort()
    
    activation_values = [activations[checkpoint] for checkpoint in checkpoint_names]
    
    plt.figure(figsize=(10, 6))
    plt.plot(activation_values, marker='o')
    plt.xticks(range(len(checkpoint_names)), checkpoint_names, rotation=45, ha='right')
    plt.xlabel('Checkpoint')
    plt.ylabel('Activation')
    plt.title(f'Longitudinal Activations for Layer: {layer_name}, Channel: {channel_num}')
    plt.grid(True)
    plt.tight_layout()
    
    if output_path:
        utils.ensure_dir_exists_for_file(output_path)
        plt.savefig(output_path)
    else:
        plt.show()


# if __name__ == '__main__':
#     from vision.featurevis import load_feature_tensor
#     # import argparse

#     # parser = argparse.ArgumentParser("Get and save longitudinal activation data")

#     image = load_feature_tensor("feature_images/resnet50/pretrained/full", 'layer2.2.conv2', 14)

#     # Generate a random noise image
#     image_size = (1, 3, 224, 224)
#     random_noise_image = torch.randn(image_size)

#     checkpoint_paths = [
#         'models/resnet50_checkpoints/1/checkpoint_2024-03-25-00h28_epoch_0_train_200.pth.tar',
#         'models/resnet50_checkpoints/1/checkpoint_2024-03-25-04h10_epoch_6_train_600.pth.tar',
#         'models/resnet50_checkpoints/1/checkpoint_2024-03-26-01h31_epoch_23_train_800.pth.tar'
#     ]
#     checkpoint_dir = 'models/resnet50_checkpoints/1/'
#     checkpoint_paths = [os.path.join(checkpoint_dir, file) for file in os.listdir(checkpoint_dir)]

#     long_activations = get_longitudinal_activations('resnet50', checkpoint_paths[0::16], 'layer2.2.conv2', 14, random_noise_image)
#     plot_longitudinal_activations(long_activations, 'layer2.2.conv2', 14, output_path='output/tmp/longitudinal_activations.png')

if __name__ == '__main__':
    from vision.featurevis import load_feature_tensor
    
    # Generate a random noise image
    image_size = (1, 3, 224, 224)  # Adjust the size according to your model's input size
    random_noise_image = torch.randn(image_size)

    checkpoint_dir = 'models/resnet50_checkpoints/1/'
    checkpoint_paths = [os.path.join(checkpoint_dir, file) for file in os.listdir(checkpoint_dir)]

    # Define the layers and channels to plot
    layers_and_channels = [
        ('layer1.0.conv1', 10),
        ('layer2.2.conv2', 14),
        ('layer3.1.conv1', 5),
        ('layer4.0.conv3', 21),
        # Add more layers and channels as needed
    ]

    for layer_name, channel_num in layers_and_channels:
        # Get activations for the random noise image
        random_activations = get_longitudinal_activations('resnet50', checkpoint_paths[0::16], layer_name, channel_num, random_noise_image)

        # Load the feature image for the current layer and channel
        feature_image = load_feature_tensor("feature_images/resnet50/pretrained/full", layer_name, channel_num)

        # Get activations for the feature image
        feature_activations = get_longitudinal_activations('resnet50', checkpoint_paths[0::16], layer_name, channel_num, feature_image)

        # Create a figure and plot both activations on the same plot
        plt.figure(figsize=(10, 6))
        
        checkpoint_names = list(random_activations.keys())
        checkpoint_names.sort()

        random_activation_values = [random_activations[checkpoint] for checkpoint in checkpoint_names]
        feature_activation_values = [feature_activations[checkpoint] for checkpoint in checkpoint_names]

        plt.plot(random_activation_values, marker='o', label='Random Image')
        plt.plot(feature_activation_values, marker='o', label='Feature Image')

        plt.xticks(range(len(checkpoint_names)), checkpoint_names, rotation=45, ha='right')
        plt.xlabel('Checkpoint')
        plt.ylabel('Activation')
        plt.title(f'Longitudinal Activations for Layer: {layer_name}, Channel: {channel_num}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        output_path = f'output/tmp/longitudinal_activations_{layer_name}_channel_{channel_num}.png'
        utils.ensure_dir_exists_for_file(output_path)
        plt.savefig(output_path)