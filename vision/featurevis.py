import os
import time
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import GaussianBlur
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import pdb
import json
from PIL import Image

def load_resnet18(weights=models.ResNet18_Weights.DEFAULT):
    model = models.resnet18(weights=weights)
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

def deprocess_image(image):
    image = image.squeeze(0).cpu().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = (image * 255).astype(np.uint8)
    return image

def activation_maximization(
    # Model & objective parameters
    model, # Passed from visualize_features
    layer_name, # Passed from visualize_features
    channel=None, # Passed from visualize_features. If None, uses all channels
    neuron=None, # Passed from visualize_features. If None, picks a neuron in the center
    aggregation='average',
    # Optimizer and convergence parameters
    max_iterations=1000,
    min_iterations=0,
    convergence_threshold=1e-3,
    convergence_window=1,
    lr_schedule=None,
    learning_rate=0.1,
    lr_decay_rate=None,
    lr_min=None,
    lr_warmup_steps=0,
    # Regularization parameters
    reg_lambda=0.01,
    use_jitter=False,
    jitter_scale=0.05,
    use_scaling=False,
    scale_range=0.1,
    use_gauss=False,
    gauss_kernel_size=3,
    use_tv_reg=False,
    tv_weight=1e-6,
    use_decorrelation=False,
    decorrelation_weight=0.1,
    # Image parameters
    input_image=None,
    feature_image_size=224,
    crop_factor=2,
    # Misc arguments
    use_gpu=None, # Set automatically by visualize_features
    **kwargs # Absorbs unnecessary **kwargs passed from visualize_features
):
    if kwargs:
        print("activation_maximization() received unnecessary arguments:", kwargs)
    assert min_iterations >= 2, "min_iterations must be at least 2"

    model.eval()
    target_layer = model
    for submodule in layer_name.split('.'):
        target_layer = target_layer._modules.get(submodule)

    channel_shape = None

    # Manage settings for individual neurons
    if aggregation == 'single':
        def get_channel_shape(module, input, output):
            nonlocal channel_shape
            channel_shape = output.shape[2:]

        temp_handle = target_layer.register_forward_hook(get_channel_shape)
        dummy_input = torch.randn(1, 3, feature_image_size, feature_image_size)
        if use_gpu:
            dummy_input = dummy_input.cuda()
            model = model.cuda()
        model(dummy_input)  # Pass a dummy input to get the channel size
        temp_handle.remove()

        height, width = channel_shape

        # Automatically select a neuron close to the center of the image if neuron is None
        if neuron is None:
            center_height, center_width = height // 2, width // 2
            neuron = (center_height, center_width)

    # Create/set up initial image
    if input_image is None:
        feature_image = torch.randn(1, 3, feature_image_size, feature_image_size, requires_grad=True)
    else:
        feature_image = input_image.clone().detach().requires_grad_(True)

    if use_gpu:
        feature_image = feature_image.cuda()
        feature_image = feature_image.detach().requires_grad_(True)
        model = model.cuda()

    feature_image_shape = feature_image.shape

    # The part where we get the activation of the channel/neuron/etc
    activation = None

    def hook(module, input, output):
        nonlocal activation
        if aggregation == 'single':
            activation = output[:, channel, neuron[0], neuron[1]]
        elif aggregation == 'average':
            activation = output[:, channel].mean()
        elif aggregation == 'sum':
            activation = output[:, channel].sum()

    handle = target_layer.register_forward_hook(hook)

    optimizer = torch.optim.Adam([feature_image], lr=learning_rate, weight_decay=reg_lambda)

    # Options for schedules for LR changing over time
    scheduler = None
    if lr_schedule == 'exponential':
        if lr_min is not None and lr_decay_rate is None:
            lr_decay_rate = (lr_min / learning_rate)**(1 / max_iterations)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_rate)
    elif lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iterations, eta_min=lr_min)
    if lr_warmup_steps > 0 and scheduler is not None:
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=lr_min / learning_rate, total_iters=lr_warmup_steps),
            scheduler
        ])

    loss_values = []
    convergence_iteration = max_iterations

    # The actual optimization process
    for i in range(max_iterations):
        # Image modifications (option) and forward pass
        if use_jitter:
            jitter = torch.randn_like(feature_image) * jitter_scale
            feature_image.data += jitter

        if use_scaling:
            scale_factor = 1 + (torch.rand(1) - 0.5) * scale_range
            scaled_size = tuple(int(s * scale_factor.item()) for s in feature_image_shape[-2:])
            feature_image = nn.functional.interpolate(feature_image, size=scaled_size, mode='bilinear', align_corners=False)

        model(feature_image)

        # Calculate loss, including frequency regularization steps
        loss = -torch.mean(activation)

        # (L2 regularization already baked into the optimizer: reg_lambda)
        if use_gauss:
            loss += GaussianBlur(kernel_size=gauss_kernel_size)(feature_image).mean()

        if use_tv_reg:
            def tv_loss(img):
                """Total variation regularization"""
                diff1 = img[:,:,1:,:] - img[:,:,:-1,:]
                diff2 = img[:,:,:,1:] - img[:,:,:,:-1]
                diff3 = img[:,:,2:,1:] - img[:,:,:-2,:-1]
                diff4 = img[:,:,1:,2:] - img[:,:,:-1,:-2]

                loss = torch.sum(torch.abs(diff1)) + torch.sum(torch.abs(diff2)) + \
                    torch.sum(torch.abs(diff3)) + torch.sum(torch.abs(diff4))

                return loss

            loss += tv_weight * tv_loss(feature_image)

        if use_decorrelation:
            # I don't think I'm implimenting this correctly yet
            def feature_decorrelation_loss(features):
                """Compute feature decorrelation loss."""
                mean_features = features.mean(dim=[2, 3], keepdim=True)
                centered_features = features - mean_features
                feature_covariance = torch.matmul(centered_features.transpose(1, 0), centered_features) / (features.size(2) * features.size(3))
                feature_correlation = feature_covariance / torch.sqrt(torch.diag(feature_covariance)[:, None] * torch.diag(feature_covariance)[None, :])
                decorrelation_loss = (feature_correlation - torch.eye(features.size(1), device=features.device)).norm(p=2)
                return decorrelation_loss

            features = target_layer.output.clone().detach()
            decorrelation_loss = feature_decorrelation_loss(features)
            loss += decorrelation_weight * decorrelation_loss

        # TODO: check if there are efficiency differences between adding to the loss
        # in various places, vs. doing it all at once at the end

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_values.append(loss.item())

        print(f"\rIteration {i+1}/{max_iterations}, Loss: {loss.item():.4f}, {layer_name}-{channel}-{neuron}", end="")

        # if i >= (min_iterations-1) and abs(loss_values[-1] - loss_values[-2]) < convergence_threshold:
        # if i >= (min_iterations-1) and max(loss_values[-convergence_window:]) - min(loss_values[convergence_window:]) < convergence_threshold:
        # Converges if the change in loss is less than convergence_threshold for the last convergence_window steps
        if i > min_iterations and max([abs(loss_values[-k] - loss_values[-k-1]) for k in range(1,convergence_window)])< convergence_threshold:
            convergence_iteration = i + 1
            break

        if scheduler is not None:
            scheduler.step()

    print()
    handle.remove()
    feature_image = deprocess_image(feature_image.cpu() if use_gpu else feature_image)
    max_activation = activation.max().item()

    # Crop the feature image to focus on the neuron
    if aggregation == 'single':
        receptive_field_size = feature_image_size // width
        crop_size = receptive_field_size * crop_factor # Adjust the crop size as needed
        crop_top = max(0, neuron[0] * receptive_field_size - crop_size // 2)
        crop_bottom = min(feature_image_size, crop_top + crop_size)
        crop_left = max(0, neuron[1] * receptive_field_size - crop_size // 2)
        crop_right = min(feature_image_size, crop_left + crop_size)
        feature_image = feature_image[crop_top:crop_bottom, crop_left:crop_right]

    return feature_image, max_activation, loss_values, convergence_iteration, layer_name, channel, neuron, aggregation

def visualize_features(model, layer_names, channels=None, neurons=None, aggregation='average',
                       crop_factor=2, checkpoint_path=None, output_path=None, batch_size=10, **kwargs):
    use_gpu = torch.cuda.is_available()
    use_all_channels = True if not channels else False

    if use_all_channels:
        channels_per_layer = [model.get_layer_output_size(layer_name) for layer_name in layer_names]
        total_batches = sum(len(range(0, channels, batch_size)) for channels in channels_per_layer)
    elif channels is None:
        raise ValueError("Please provide either the 'channels' argument or set 'use_all_channels' to True.")
    else:
        total_batches = len(layer_names) * len(range(0, len(channels), batch_size))

    feature_images = []
    start_time = time.time()
    processed_batches = 0

    if neurons is None:
        neurons = [None]

    for layer_idx, layer_name in enumerate(layer_names):
        if use_all_channels:
            channels = list(range(channels_per_layer[layer_idx]))

        for i in range(0, len(channels), batch_size):
            batch_channels = channels[i:i+batch_size]
            batch_feature_images = []

            for channel in batch_channels:
                for neuron in neurons:
                    input_image = None
                    if checkpoint_path is not None:
                        input_image = load_feature_image(layer_name, channel, neuron, aggregation, checkpoint_path)
                        input_image = torch.from_numpy(input_image).float()
                        input_image = input_image.permute(2, 0, 1)  # Rearrange dimensions to [channels, height, width]
                        input_image = input_image.unsqueeze(0) # Add batch dimension

                    feature_image, activation, loss_values, convergence_iteration, _, _, neuron, aggregation = activation_maximization(
                        model, layer_name, channel, neuron, input_image=input_image, **kwargs
                    )
                    batch_feature_images.append((feature_image, activation, loss_values, convergence_iteration, layer_name, channel, neuron, aggregation))

            feature_images.extend(batch_feature_images)

            if output_path is not None:
                save_feature_images(batch_feature_images, output_path)

            processed_batches += 1
            elapsed_time = time.time() - start_time
            estimated_total_time = (elapsed_time / processed_batches) * total_batches
            remaining_time = estimated_total_time - elapsed_time

            print(f"Layer: {layer_name}, Batch: {processed_batches}/{total_batches}, "
                  f"Elapsed Time: {elapsed_time:.2f}s, Estimated Remaining Time: {remaining_time:.2f}s")

    return feature_images

def plot_feature_images(feature_images, num_columns=5):
    num_images = len(feature_images)
    num_rows = (num_images + num_columns - 1) // num_columns

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns*3, num_rows*3))
    axes = axes.flatten()

    for i, (image, activation, _, _, layer_name, channel, neuron, aggregation) in enumerate(feature_images):
        axes[i].imshow(image)
        axes[i].axis('off')
        if aggregation == 'single':
            axes[i].set_title(f"Layer: {layer_name}\nChannel: {channel}\nNeuron: {neuron}\nAggregation: {aggregation}\nActivation: {activation:.2f}")
        else:
            axes[i].set_title(f"Layer: {layer_name}\nChannel: {channel}\nAggregation: {aggregation}\nActivation: {activation:.2f}")

    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def feature_image_paths(directory, layer_name, channel, neuron, aggregation):
    if neuron is not None:
        filename = f"{directory}/layer_{layer_name}_channel_{channel}_neuron_{neuron[0]}_{neuron[1]}.jpg"
    elif aggregation == 'single':
        filename = f"{directory}/layer_{layer_name}_channel_{channel}_center.jpg"
    else:
        filename = f"{directory}/layer_{layer_name}_channel_{channel}_{aggregation}.jpg"

    info_filename = filename[:-4] + ".info.json"

    return filename, info_filename

def save_feature_images(feature_images, checkpoint_path, **kwargs):
    directory = f"{checkpoint_path}"
    os.makedirs(directory, exist_ok=True)
    print(f"save_feature_images called with checkpoint_path={checkpoint_path}")

    for image, activation, loss_values, convergence_iteration, layer_name, channel, neuron, aggregation in feature_images:
        filename, info_filename = feature_image_paths(directory, layer_name, channel, neuron, aggregation)
        plt.imsave(filename, image)
        info_data = {
            "Layer": layer_name,
            "Channel": channel,
            "Neuron": neuron,
            "Activation": activation,
            "Convergence Iteration": convergence_iteration,
            "Loss Values": loss_values,
            "Parameters": kwargs
        }
        with open(info_filename, 'w') as file:
            json.dump(info_data, file)

def load_feature_image(layer_name, channel, neuron, aggregation, checkpoint_path):
    filename, _ = feature_image_paths(checkpoint_path, layer_name, channel, neuron, aggregation)
    image = plt.imread(filename)
    return image

def plot_feature_images_from_checkpoint(checkpoint_path, layer_names, channels, neurons, aggregation, num_columns=5, batch_size=10):
    feature_images = []

    for layer_name in layer_names:
        for i in range(0, len(channels), batch_size):
            batch_channels = channels[i:i+batch_size]
            batch_feature_images = []

            for channel in batch_channels:
                for neuron in neurons:
                    image = load_feature_image(layer_name, channel, neuron, aggregation, checkpoint_path)
                    _, info_filename = feature_image_paths(checkpoint_path, layer_name, channel, neuron, aggregation)
                    with open(info_filename, 'r') as file:
                        info_data = json.load(file)
                        activation = info_data["Activation"]
                        convergence_iteration = info_data["Convergence Iteration"]
                        loss_values = info_data["Loss Values"]

                    batch_feature_images.append((image, activation, loss_values, convergence_iteration, layer_name, channel, neuron))

            feature_images.extend(batch_feature_images)
            plot_feature_images(batch_feature_images, num_columns)

    return feature_images
