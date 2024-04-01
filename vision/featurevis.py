import os
import time
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import pdb
import json

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

def activation_maximization(model, layer_name, channel, max_iterations, min_iterations,
                            lr_schedule, learning_rate, lr_decay_rate, lr_min, lr_warmup_steps,
                            reg_lambda, convergence_threshold, use_gpu, input_image=None):
    model.eval()
    target_layer = model
    for submodule in layer_name.split('.'):
        target_layer = target_layer._modules.get(submodule)

    if input_image is None:
        initial_image = torch.randn(1, 3, 224, 224, requires_grad=True)
    else:
        initial_image = input_image.clone().detach().requires_grad_(True)

    if use_gpu:
        initial_image = initial_image.cuda()
        initial_image = initial_image.detach().requires_grad_(True)
        model = model.cuda()

    activation = None

    def hook(module, input, output):
        nonlocal activation
        activation = output[:, channel]

    handle = target_layer.register_forward_hook(hook)
    optimizer = torch.optim.Adam([initial_image], lr=learning_rate, weight_decay=reg_lambda)

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
    # else:
    #     scheduler = None

    loss_values = []
    convergence_iteration = max_iterations

    for i in range(max_iterations):
        model(initial_image)
        loss = -torch.mean(activation)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_values.append(loss.item())

        # Progress bar updates
        print(f"\rIteration {i+1}/{max_iterations}, Loss: {loss.item():.4f}, {layer_name}-{channel}", end="")

        # Check for convergence
        if i > min_iterations and abs(loss_values[-1] - loss_values[-2]) < convergence_threshold:
            convergence_iteration = i + 1
            break

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

    print()
    handle.remove()
    feature_image = deprocess_image(initial_image.cpu() if use_gpu else initial_image)
    max_activation = activation.max().item()

    return feature_image, max_activation, loss_values, convergence_iteration, layer_name, channel

def plot_feature_images(feature_images, num_columns=5):
    num_images = len(feature_images)
    num_rows = (num_images + num_columns - 1) // num_columns

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns*3, num_rows*3))
    axes = axes.flatten()

    for i, (image, activation, _, _, layer_name, channel) in enumerate(feature_images):
        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(f"Lyr:{layer_name}-Chn:{channel}@Act:{activation:.2f}")

    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def save_feature_images(feature_images, checkpoint_path):
    directory = f"{checkpoint_path}"
    os.makedirs(directory, exist_ok=True)
    print(f"save_feature_images called with checkpoint_path={checkpoint_path}")

    for image, activation, loss_values, convergence_iteration, layer_name, channel in feature_images:
        filename = f"{directory}/layer_{layer_name}_channel_{channel}.jpg"
        plt.imsave(filename, image)

        # Save additional information as JSON
        info_filename = f"{directory}/layer_{layer_name}_channel_{channel}_info.json"
        info_data = {
            "Layer": layer_name,
            "Channel": channel,
            "Activation": activation,
            "Convergence Iteration": convergence_iteration,
            "Loss Values": loss_values
        }
        with open(info_filename, 'w') as file:
            json.dump(info_data, file)

def load_feature_image(layer_name, channel, checkpoint_path):
    directory = f"{checkpoint_path}"
    filename = f"{directory}/layer_{layer_name}_channel_{channel}.jpg"
    image = plt.imread(filename)
    return image

import time

def visualize_features(model, layer_names, channels=None, max_iterations=100, min_iterations=50,
                       lr_schedule=None, learning_rate=0.1, lr_decay_rate=None, lr_min=None, lr_warmup_steps=0,
                       reg_lambda=0.01, convergence_threshold=1e-4, use_all_channels=False,
                       checkpoint_path=None, output_path=None, batch_size=10):
    use_gpu = torch.cuda.is_available()

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

    for layer_idx, layer_name in enumerate(layer_names):
        if use_all_channels:
            channels = list(range(channels_per_layer[layer_idx]))

        for i in range(0, len(channels), batch_size):
            batch_channels = channels[i:i+batch_size]
            batch_feature_images = []

            for channel in batch_channels:
                input_image = None
                if checkpoint_path is not None:
                    input_image = load_feature_image(layer_name, channel, checkpoint_path)
                    input_image = preprocess_image(input_image)

                feature_image, activation, loss_values, convergence_iteration, _, _ = activation_maximization(
                    model, layer_name, channel, max_iterations, min_iterations,
                    lr_schedule, learning_rate, lr_decay_rate, lr_min, lr_warmup_steps,
                    reg_lambda, convergence_threshold, use_gpu, input_image
                )
                batch_feature_images.append((feature_image, activation, loss_values, convergence_iteration, layer_name, channel))

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

def plot_feature_images_from_checkpoint(checkpoint_path, layer_names, channels, num_columns=5, batch_size=10):
    feature_images = []

    for layer_name in layer_names:
        for i in range(0, len(channels), batch_size):
            batch_channels = channels[i:i+batch_size]
            batch_feature_images = []

            for channel in batch_channels:
                image = load_feature_image(layer_name, channel, checkpoint_path)
                info_filename = f"{checkpoint_path}/layer_{layer_name}_channel_{channel}_info.json"
                with open(info_filename, 'r') as file:
                    info_data = json.load(file)
                    activation = info_data["Activation"]
                    convergence_iteration = info_data["Convergence Iteration"]
                    loss_values = info_data["Loss Values"]

                batch_feature_images.append((image, activation, loss_values, convergence_iteration, layer_name, channel))

            feature_images.extend(batch_feature_images)
            plot_feature_images(batch_feature_images, num_columns)

    return feature_images
