import json
from functools import partial
import matplotlib.pyplot as plt
from vision.featurevis import load_torchvision_model, load_feature_image

def register_activation_hooks(model, loci_dict):
    activations = {}
    
    def hook_factory(layer_name, index):
        def hook(module, input, output):
            activations[(layer_name, index)] = output[0, index].item()
        return hook
    
    for layer_name, neuron_channel_indices in loci_dict.items():
        layer = model
        for submodule in layer_name.split('.'):
            layer = layer._modules.get(submodule)
        for index in neuron_channel_indices:
            layer.register_forward_hook(hook_factory(layer_name, index))
    
    return activations

def get_longitudinal_activation(feature_images, model_name, checkpoint_paths, loci_dict, output_path=None):
    longitudinal_data = []
    
    for checkpoint_path in checkpoint_paths:
        model = load_torchvision_model(model_name, checkpoint_path)
        model.eval()
        
        activations = register_activation_hooks(model, loci_dict)
        
        checkpoint_activations = []
        for image in feature_images:
            model(image)
            checkpoint_activations.append(activations.copy())
        
        longitudinal_data.append((checkpoint_path, checkpoint_activations))
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(longitudinal_data, f)
    
    return longitudinal_data

def plot_longitudinal_activation(longitudinal_data, loci_dict):
    checkpoints = [data[0] for data in longitudinal_data]
    activations = [data[1] for data in longitudinal_data]
    
    plt.figure(figsize=(10, 6))
    for layer_name, neuron_channel_indices in loci_dict.items():
        for index in neuron_channel_indices:
            activation_values = [checkpoint_activations[(layer_name, index)] for checkpoint_activations in activations]
            plt.plot(checkpoints, activation_values, label=f"Layer: {layer_name}, Index: {index}")
    
    plt.xlabel("Model Checkpoint")
    plt.ylabel("Activation")
    plt.title("Longitudinal Activation")
    plt.legend()
    plt.show()

if __name__ == 'main':
    image = load_feature_image("feature_images/resnet50/pretrained/full", 'layer3.3.conv2', 22)
    x = get_longitudinal_activation([image], 
                                    'resnet50', 
                                    ['models/resnet50_checkpoints/1/checkpoint_2024-03-25-00h28_epoch_0_train_200.pth.tar',
                                                          'models/resnet50_checkpoints/1/checkpoint_2024-03-25-04h10_epoch_6_train_600.pth.tar', 
                                                          'models/resnet50_checkpoints/1/checkpoint_2024-03-26-01h31_epoch_23_train_800.pth.tar'], 
                                    {'layer2.1.conv1': [0,1], 'layer2.3.conv3': [33]})
