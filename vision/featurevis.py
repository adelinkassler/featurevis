import os, sys, time, copy
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import GaussianBlur
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import pdb
import json, re, random
from PIL import Image
import submitit
from itertools import product
from functools import partial

torch.autograd.set_detect_anomaly(True)

def load_torchvision_model(model_name, checkpoint_path=None, device='cpu', verbose=True):
    # Loads a torchvision model using default weights or a checkpoint file
    if hasattr(models, model_name):
        model_class = getattr(models, model_name)
    else:
        raise ValueError(f"Could not find model torchvision.models.{model_name}")
    
    if checkpoint_path is None:
        weights_class = None
        for attr_name in dir(models):
            if attr_name.upper().startswith(model_name.upper()) and attr_name.endswith('_Weights'):
                weights_class = getattr(models, attr_name)
                break
        
        if weights_class is None:
            raise ValueError(f"Could not find weights for {model_name} in torchvision.models")

        model = model_class(weights=weights_class.DEFAULT)
        if verbose:
            print(f"Loaded {model_name} from torchvision with default weights")
    else:
        model = model_class()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if verbose:
            print(f"Loaded {model_name} from torchvision and checkpoint at {checkpoint_path}")

    model.to(device)
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
    model=None, # Passed from visualize_features
    model_path=None, # Alternative to model
    layer_name=None, # Passed from visualize_features
    channel=None, # Passed from visualize_features. If None, uses all channels
    neuron=None, # Passed from visualize_features. If None, picks a neuron in the center
    aggregation='average',
    number_of_images=1, # Number of images to generate
    diversity_weight=1.0, # Penalty on cosine similarity
    prev_feature_images=None,
    # Optimizer and convergence parameters
    max_iterations=1000,
    min_iterations=2,
    convergence_threshold=1e-4,
    convergence_window=2,
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
    use_gpu=None, # Set automatically if argument is None
    progress_bar=True,
    seed=None
):
    if seed is not None:
        random.seed(seed)
    # Check that arguments are all legal
    if model is None:
        if model_path is not None:
            model = torch.load(model_path)
        else:
            raise ValueError("model or model_path must be provided")
    min_iterations = max(min_iterations, convergence_window, lr_warmup_steps)
    assert convergence_window > 1, "convergence_window must be at least 2"

    # Find device type, create device object for later use
    if use_gpu is None or use_gpu is True:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        elif use_gpu is True:
            raise RuntimeError("use_gpu is set to True, but neither CUDA nor MPS are available on this system")
        else:
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    
    # Doing this keeps the info files for the output more readable
    if not use_jitter:
        jitter_scale = None
    if not use_scaling:
        scale_range = None
    if not use_gauss:
        gauss_kernel_size = None
    if not use_tv_reg:
        tv_weight = None
    if not use_decorrelation:
        decorrelation_weight = None

    model = model.to(device)
    prev_feature_image = [image.to(device) for image in prev_feature_images]

    # Find target layer
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
        if device.type == 'cuda':
            dummy_input = dummy_input.cuda()
        elif device.type == 'mps':
            dummy_input.to(device)
        model(dummy_input)  # Pass a dummy input to get the channel size
        temp_handle.remove()

        height, width = channel_shape

        # Automatically select a neuron close to the center of the image if neuron is None
        if neuron is None:
            center_height, center_width = height // 2, width // 2
            neuron = (center_height, center_width)

    # Define and register hook function for target neuron/channel activation
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

    # Create/set up initial image
    if input_image is None:
        feature_image = torch.randn(1, 3, feature_image_size, feature_image_size, requires_grad=True)
    else:
        feature_image = input_image.clone().detach().requires_grad_(True)

    feature_image_shape = feature_image.shape
    
    # Configure image on device
    feature_image = feature_image.to(device)
    feature_image = feature_image.detach().requires_grad_(True)

    # Create the optimizer we will use
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

    # Inner optimzation loop (per feature image)
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
        # pdb.set_trace()
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

        # Diversity penalty
        if prev_feature_images and diversity_weight > 0:
            cosine_similarities = []
            for prev_feature_image in prev_feature_images:
                # prev_feature_image = prev_feature_image.to(device)
                cosine_similarity = torch.nn.functional.cosine_similarity(
                    feature_image.flatten(), prev_feature_image.flatten(), dim=0
                )
                cosine_similarities.append(cosine_similarity)
            diversity_penalty = torch.mean(torch.stack(cosine_similarities))
            loss += diversity_weight * diversity_penalty

        # TODO: check if there are efficiency differences between adding to the loss
        # in various places, vs. doing it all at once at the end

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        loss_values.append(loss.item())

        if progress_bar:
            print(f"\rIteration {i+1}/{max_iterations}, Loss: {loss.item():.4f}, {layer_name}-{channel}-{neuron}", end="", flush=True)

        # if i >= (min_iterations-1) and abs(loss_values[-1] - loss_values[-2]) < convergence_threshold:
        # if i > min_iterations and max([abs(loss_values[-k] - loss_values[-k-1]) for k in range(1,convergence_window+1)]) < convergence_threshold:
        # Converges if the change in loss is less than convergence_threshold for the last convergence_window steps
        if i >= (min_iterations-1) and max(loss_values[-convergence_window:]) - min(loss_values[-convergence_window:]) < convergence_threshold:
            convergence_iteration = i + 1
            break

        if scheduler is not None:
            scheduler.step()

    print(flush=True)
    handle.remove()
    # feature_image = deprocess_image(feature_image.cpu() if device != 'cpu' else feature_image)
    max_activation = activation.max().item()

    # Crop the feature image to focus on the neuron
    # if aggregation == 'single':
    #     receptive_field_size = feature_image_size // width
    #     crop_size = receptive_field_size * crop_factor
    #     crop_top = max(0, neuron[0] * receptive_field_size - crop_size // 2)  
    #     crop_bottom = min(feature_image_size, crop_top + crop_size)
    #     crop_left = max(0, neuron[1] * receptive_field_size - crop_size // 2) 
    #     crop_right = min(feature_image_size, crop_left + crop_size)
    #     feature_image = feature_image[:, :, crop_top:crop_bottom, crop_left:crop_right]

    return feature_image, max_activation, loss_values, convergence_iteration, layer_name, channel, neuron, aggregation

def activation_maximization_batch(job_args):
    model, layer_name, channels, neurons, aggregation, output_path, image_loader, number_of_images, kwargs = job_args
    print(f"\nStarting job for layer: {layer_name}, channels: {channels}, neurons: {neurons}, aggregation: {aggregation}",
          f"{number_of_images} images per feature. Additional arguments: {kwargs}")
    job_results = []
    for channel, neuron in zip(channels, neurons):
        images, activations, loss_values_lists, convergence_iterations = [], [], [], []
        prev_feature_images = []
        for _ in range(number_of_images):
            input_image = image_loader(layer_name, channel, neuron, aggregation) if image_loader is not None else None
            print(f"\nProcessing channel {channel} and neuron {neuron} for layer {layer_name}")
            result = activation_maximization(model=model, layer_name=layer_name, channel=channel,
                                             neuron=neuron, aggregation=aggregation,
                                             input_image=input_image, prev_feature_images=prev_feature_images, **kwargs)
            images.append(deprocess_image(result[0]))
            activations.append(result[1])
            loss_values_lists.append(result[2])
            convergence_iterations.append(result[3])
            prev_feature_images.append(result[0])
            save_feature_images([(images, activations, loss_values_lists, convergence_iterations, layer_name, channel, neuron, aggregation)], output_path, kwargs)
            print(f"Saved feature image for channel {channel} and neuron {neuron}")
    return job_results

def visualize_features(model, layer_names=None, channels=None, neurons=None, aggregation='average',
                       init_image_loader=None, number_of_images=1, output_path=None, batch_size=10, use_gpu=None,
                       parallel=False, return_output=True, cleanup=False, **kwargs):
    use_all_channels = True if channels is None else False

    if layer_names is None:
        layer_names = [name for name, layer in model.named_modules() if isinstance(layer, nn.Conv2d)]
    elif isinstance(layer_names, str):
        layer_names = [layer_names]

    if use_all_channels:
        print("Using all convolutional channels")
        channels_per_layer = {}
        for layer_name in layer_names:
            target_layer = model
            for submodule in layer_name.split('.'):
                target_layer = target_layer._modules.get(submodule)
            
            if isinstance(target_layer, nn.Conv2d):
                num_channels = target_layer.out_channels
            elif isinstance(target_layer, nn.BatchNorm2d):
                num_channels = target_layer.num_features
            elif isinstance(target_layer, nn.Linear):
                num_channels = target_layer.out_features
            elif isinstance(target_layer, nn.Sequential):
                conv_layers = [layer for name, layer in model.named_modules() if isinstance(layer, nn.Conv2d)\
                               and name[:len(layer_name)] == layer_name]
                if conv_layers:
                    num_channels = conv_layers[-1].out_channels
                else:
                    raise ValueError(f"No convolutional layers found in the sequential block: {layer_name}")
            else:
                raise ValueError(f"Unsupported layer type: {type(target_layer)}")
            
            channels_per_layer[layer_name] = num_channels
        
        total_batches = sum(len(range(0, channels, batch_size)) for channels in channels_per_layer.values())
    elif channels is None:
        raise ValueError("Please provide either the 'channels' argument or set 'use_all_channels' to True.")
    else:
        total_batches = len(layer_names) * len(range(0, len(channels), batch_size))

    feature_images = []
    start_time = time.time()
    processed_batches = 0

    # Create executor
    executor = submitit.AutoExecutor(folder=output_path)
    if parallel:
        executor.update_parameters(timeout_min=10, slurm_partition='single', gpus_per_node=1)
    else:
        #FIXME: local execution is b0rked, and maybe never worked in the first place
        executor.update_parameters(timeout_min=10)
    #     executor.update_parameters(timeout_min=10, local_cpus=1, local_gpus=1 if use_gpu else 0)
        
    # Create arguments for each job  
    job_args = []
    for layer_name in layer_names:
        if channels is None:
            channels = list(range(channels_per_layer[layer_name]))
        neurons = neurons if neurons else [None]

        channels_neurons = list(product(channels, neurons))
        for i in range(0, len(channels_neurons), batch_size):
            job_channels = [c for c, _ in channels_neurons[i:i+batch_size]]
            job_neurons = [n for _, n in channels_neurons[i:i+batch_size]]
            job_args.append((model, layer_name, job_channels, job_neurons, aggregation, output_path, 
                             init_image_loader, number_of_images, kwargs))
    
    # Submit jobs
    job_array = executor.map_array(activation_maximization_batch, job_args)
    if parallel:
        print(f"Job array submitted with ID: {job_array[0].job_id.split('_')[0]}")
        track_job_array_progress([job_array])
    else:
        track_local_job_progress([job_array])

    # Collect results
    job_results = [job.result() for job in job_array]
    for job_result in job_results:
        feature_images.extend(job_result)
        save_feature_images(job_result, output_path, kwargs)

        processed_batches += 1
        elapsed_time = time.time() - start_time
        estimated_total_time = (elapsed_time / processed_batches) * total_batches
        remaining_time = estimated_total_time - elapsed_time
        print(f"Batch: {processed_batches}/{total_batches}, "
              f"Elapsed Time: {elapsed_time:.2f}s, Estimated Remaining Time: {remaining_time:.2f}s", 
              flush=True)

    if return_output:
        return feature_images

def track_job_array_progress(job_arrays):
    print("Waiting for all jobs to complete...")
    num_jobs = sum(len(job_array) for job_array in job_arrays)
    jobs_ended = 0
    status_msg_len = 0
    while jobs_ended < num_jobs:
        job_stats = {
            'PENDING': 0,
            'RUNNING': 0,
            'COMPLETED': 0,
            'FAILED': 0,
            'CANCELLED': 0,
            'TIMEOUT': 0,
            'NODE_FAIL': 0,
            'UNKNOWN': 0
        }
        for job_array in job_arrays:
            for job in job_array:
                job_stats[job.state] += 1
        jobs_ended = job_stats['COMPLETED'] + job_stats['CANCELLED'] + job_stats['FAILED'] + job_stats['TIMEOUT']
        progress_bar = "[{0}{1}] {2}/{3} jobs finished".format(
            "=" * (jobs_ended * 30 // num_jobs),
            " " * ((num_jobs - jobs_ended) * 30 // num_jobs),
            jobs_ended,
            num_jobs
        )
        state_count_nonzero = []
        state_count_labels = {'COMPLETED':'Completed', 'RUNNING':'Running', 'PENDING':'Pending', 'FAILED':'Failed',
                              'CANCELLED':'Cancelled', 'TIMEOUT':'Timeout', 'NODE_FAIL':'Node Failure', 'UNKNOWN':'Unknown'}
        for state in job_stats.keys():
            if job_stats[state] > 0:
                state_count_nonzero.append(f"{state_count_labels[state]}: {job_stats[state]}")
        status_msg = f"\r{progress_bar} ({', '.join(state_count_nonzero)})"
        print(status_msg + " "*max(status_msg_len-len(status_msg), 0), end="", flush=True)
        status_msg_len = len(status_msg)
        time.sleep(1)  # Wait for 1 second before updating the progress
    print("\nAll jobs completed.")
    if jobs_ended != job_stats['COMPLETED']:
        print("Warning: some jobs finished with nonzero exit status", file=sys.stderr)

def track_local_job_progress(jobs: List[submitit.Job], poll_frequency: float = 1.0):
    num_jobs = len(jobs)
    completed_jobs = 0
    start_time = time.time()

    while completed_jobs < num_jobs:
        completed_jobs = sum(job.done() for job in jobs)
        elapsed_time = time.time() - start_time
        progress_bar = f"[{'#' * (completed_jobs * 20 // num_jobs):20}]"
        print(f"\r{progress_bar} {completed_jobs}/{num_jobs} jobs completed "
              f"(Elapsed: {elapsed_time:.1f}s)", end="", flush=True)
        time.sleep(poll_frequency)

    print(f"\nAll {num_jobs} jobs completed.")

def plot_feature_images(feature_images, num_columns=5, output_path=None):
    num_images = len(feature_images)
    num_rows = (num_images + num_columns - 1) // num_columns

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns*3, num_rows*3))
    axes = axes.flatten()

    for i, (image, activation, _, _, layer_name, channel, neuron, aggregation) in enumerate(feature_images): # add aggregation back
        axes[i].imshow(image)
        axes[i].axis('off')
        if aggregation == 'single':
            axes[i].set_title(f"Layer: {layer_name}\nChannel: {channel}\nNeuron: {neuron}\nAggregation: {aggregation}\nActivation: {activation:.2f}")
        else:
            axes[i].set_title(f"Layer: {layer_name}\nChannel: {channel}\nAggregation: {aggregation}\nActivation: {activation:.2f}")

    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()

def feature_image_paths(directory, layer_name, channel, neuron=None, aggregation="average", image_num=None):
    image_num_suffix = "" if image_num is None else '-' + str(image_num)
    if neuron is not None:
        filename = f"{directory}/layer_{layer_name}_channel_{channel}_neuron_{neuron[0]}_{neuron[1]}{image_num_suffix}.jpg"
    elif aggregation == 'single':
        filename = f"{directory}/layer_{layer_name}_channel_{channel}_center{image_num_suffix}.jpg"
    else:
        filename = f"{directory}/layer_{layer_name}_channel_{channel}_{aggregation}{image_num_suffix}.jpg"

    info_filename = filename[:-4] + ".info.json"

    return filename, info_filename

def save_feature_images(feature_images, output_path, params):
    if output_path is None:
        print("No output path provided; feature images have not been saved")
        return
    directory = f"{output_path}"
    os.makedirs(directory, exist_ok=True)
    print(f"save_feature_images called with output_path={output_path}")

    for images, activations, loss_values_lists, convergence_iterations, layer_name, channel, neuron, aggregation in feature_images:
        for j, (image, activation, loss_values, convergence_iteration) in enumerate(zip(images, activations, loss_values_lists, convergence_iterations)):
            filename, info_filename = feature_image_paths(directory, layer_name, channel, neuron, aggregation, j)
            plt.imsave(filename, image)
            info_data = {
                "Layer": layer_name,
                "Channel": channel,
                "Neuron": neuron,
                "Aggregation": aggregation,
                "Image Number": j,
                "Activation": activation,
                "Convergence Iteration": convergence_iteration,
                "Parameters": params,
                "Loss Values": loss_values
            }
            with open(info_filename, 'w') as file:
                json.dump(info_data, file, indent=2)

def load_feature_image(image_dir, layer_name, channel, neuron=None, aggregation='average', image_num=None):
    filename, _ = feature_image_paths(image_dir, layer_name, channel, neuron, aggregation, image_num)
    image = plt.imread(filename)
    return image

def preprocess_stored_feature_image(image):
    assert isinstance(image, np.ndarray), "Input image is not a numpy array"
    # Preprocess feature image for use as an input image to activation_maximization
    image = torch.from_numpy(image).float()
    image = image.permute(2, 0, 1)  # Rearrange dimensions to [channels, height, width]
    image = image.unsqueeze(0) # Add batch dimension
    return image

def load_feature_tensor(image_dir, layer_name, channel, neuron=None, aggregation='average', image_num=None, device='cpu'):
    image = load_feature_image(image_dir, layer_name, channel, neuron, aggregation, image_num)
    image = preprocess_stored_feature_image(image)
    image.to(device)
    return image

def make_feature_image_loader_with_fallback(image_dirs):
    def image_loader(layer_name, channel, neuron, aggregation):
        assert isinstance(image_dirs, list) and isinstance(image_dirs[0], str), \
            "Incorrectly formatted argument to image loader"
        i = 0
        image = None
        for path in reversed(image_dirs):
            try:
                i += 1
                image = load_feature_image(path, layer_name, channel, neuron, aggregation)
                image = preprocess_stored_feature_image(image)
                break
            except:
                next
        if image is None:
            print("Could not find a feature image in any of the provided paths")
        elif i>1:
            print(f"Could not load feature image from {image_dirs[-1]}, going back {i} checkpoints to {image_dirs[-i]}")
        return image
    return image_loader

def plot_saved_feature_images(images_dir, layer_names, channels, neurons, aggregation, num_columns=5, batch_size=10, output_path=None):
    feature_images = []

    for layer_name in layer_names:
        for i in range(0, len(channels), batch_size):
            batch_channels = channels[i:i+batch_size]
            batch_feature_images = []

            for channel in batch_channels:
                for neuron in neurons:
                    image = load_feature_image(images_dir, layer_name, channel, neuron, aggregation)
                    _, info_filename = feature_image_paths(images_dir, layer_name, channel, neuron, aggregation)
                    with open(info_filename, 'r') as file:
                        info_data = json.load(file)
                        activation = info_data["Activation"]
                        convergence_iteration = info_data["Convergence Iteration"]
                        loss_values = info_data["Loss Values"]

                    batch_feature_images.append((image, activation, loss_values, convergence_iteration, layer_name, channel, neuron))

            feature_images.extend(batch_feature_images)
            plot_feature_images(batch_feature_images, num_columns, output_path)

    return feature_images

def load_info_data(checkpoint_dirs, layer_name, channel_num, info_key="Activation", aggregation='average'):
    data = []
    for checkpoint_dir in checkpoint_dirs:
        _, info_file = feature_image_paths(checkpoint_dir, layer_name, channel_num, None, aggregation)
        with open(info_file, 'r') as f:
            info_data = json.load(f)
            data.append(info_data[info_key])
    return data

def load_info_data(checkpoint_dirs, layer_name, channel_num, info_key, aggregation='average'):
    data = []
    for checkpoint_dir in checkpoint_dirs:
        _, info_file = feature_image_paths(checkpoint_dir, layer_name, channel_num, None, aggregation)
        with open(info_file, 'r') as f:
            info_data = json.load(f)
            data.append(info_data[info_key])
    return data

def plot_info_over_training(checkpoint_dirs_path, layer_name, channel_num, info_key, aggregation='average', output_dir=None):
    checkpoint_dirs = [d for d in os.listdir(checkpoint_dirs_path)
                       if os.path.isdir(os.path.join(checkpoint_dirs_path, d))]
    checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(re.search(r'train_(\d+)', x).group(1)))

    data = load_info_data(checkpoint_dirs, layer_name, channel_num, info_key, aggregation)

    plt.figure(figsize=(10, 5))
    plt.plot(data)
    plt.xlabel('Training Iteration')
    plt.ylabel(info_key)
    plt.title(f'{info_key} vs Training for Layer {layer_name} Channel {channel_num}')
    plt.tight_layout()

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{info_key.lower()}_layer_{layer_name}_channel_{channel_num}.png'))

    plt.show()

def plot_feature_images_over_training(checkpoint_dirs_path, layer_name, channel_num, aggregation='average', output_dir=None):
    checkpoint_dirs = [d for d in os.listdir(checkpoint_dirs_path)
                       if os.path.isdir(os.path.join(checkpoint_dirs_path, d))]
    checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(re.search(r'train_(\d+)', x).group(1)))

    fig, axes = plt.subplots(2, len(checkpoint_dirs) // 2, figsize=(20, 5))
    for i, checkpoint_dir in enumerate(checkpoint_dirs):
        img_file, _ = feature_image_paths(os.path.join(checkpoint_dirs_path, checkpoint_dir),
                                          layer_name, channel_num, None, aggregation)
        img = Image.open(img_file)
        axes.flat[i].imshow(img)
        axes.flat[i].set_title(checkpoint_dir)
        axes.flat[i].set_xticks([])
        axes.flat[i].set_yticks([])
    plt.tight_layout()

    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, f'feature_images_layer_{layer_name}_channel_{channel_num}.png'))

    plt.show()

def get_channel_influences(model, target_layer, target_channel, influence_layer):
    # Get the layers in the model
    layers = list(model.named_modules())

    # Find the target and influence layer indices
    target_layer_index = [i for i, (name, module) in enumerate(layers) if name == target_layer]
    influence_layer_index = [i for i, (name, module) in enumerate(layers) if name == influence_layer]

    if not target_layer_index:
        raise ValueError(f"Target layer '{target_layer}' not found in the model.")
    if not influence_layer_index:
        raise ValueError(f"Influence layer '{influence_layer}' not found in the model.")

    target_layer_index = target_layer_index[0]
    influence_layer_index = influence_layer_index[0]

    # Ensure the influence layer comes before the target layer
    if influence_layer_index >= target_layer_index:
        raise ValueError("The influence layer must come before the target layer in the model.")

    # Get the number of channels in the influence layer
    influence_channels = layers[influence_layer_index][1].out_channels

    # Initialize influence scores for each channel in the influence layer
    influence_scores = [0] * influence_channels

    # Iterate through the paths from the influence layer to the target layer
    path_start = influence_layer_index
    while path_start < target_layer_index:
        layer = layers[path_start][1]

        if isinstance(layer, nn.Sequential):
            # Recursively process the sublayers within the Sequential layer
            for sublayer in layer:
                if isinstance(sublayer, nn.Conv2d):
                    # Get the weights connecting the current sublayer to the next layer
                    weights = sublayer.weight

                    # Calculate the influence of each channel in the current sublayer on the target channel
                    for channel in range(weights.shape[1]):
                        influence_scores[channel] += weights[target_channel, channel].sum().item()

                elif isinstance(sublayer, (nn.ReLU, nn.BatchNorm2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
                    # These sublayers don't affect the influence scores
                    pass

                else:
                    # Handle custom module types recursively
                    for module in sublayer.modules():
                        if isinstance(module, nn.Conv2d):
                            # Get the weights connecting the current module to the next layer
                            weights = module.weight

                            # Calculate the influence of each channel in the current module on the target channel
                            for channel in range(weights.shape[1]):
                                influence_scores[channel] += weights[target_channel, channel].sum().item()

            # Update the path_start index to skip the processed sublayers
            path_start += len(layer)

        elif isinstance(layer, nn.Conv2d):
            # Get the weights connecting the current layer to the next layer
            weights = layer.weight

            # Calculate the influence of each channel in the current layer on the target channel
            for channel in range(weights.shape[1]):
                influence_scores[channel] += weights[target_channel, channel].sum().item()

            path_start += 1

        elif isinstance(layer, (nn.ReLU, nn.BatchNorm2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
            # These layers don't affect the influence scores
            path_start += 1

        else:
            raise ValueError(f"Unsupported layer type: {type(layer).__name__}")

    # Combine the results into a list of tuples (channel, influence_score)
    channel_influences = list(enumerate(influence_scores))

    # Sort the channels by their influence scores in descending order
    channel_influences.sort(key=lambda x: x[1], reverse=True)

    return channel_influences

def find_strongest_influences(model, layer_a, layer_b, channel_num):
    module_b = eval(f"model.{layer_b}", model.__dict__)
    module_a = eval(f"model.{layer_a}", model.__dict__)
    
    num_channels_a = module_a.out_channels
    
    total_effects = torch.zeros(num_channels_a)
    path_counts = torch.zeros(num_channels_a)
    
    current_layer = None
    
    for name, module in model.named_modules():
        if name == layer_a:
            current_layer = module
            path_weights = torch.eye(num_channels_a)
        elif current_layer is not None:
            if isinstance(module, torch.nn.Conv2d):
                norm = torch.linalg.norm(module.weight)
                w = module.weight / norm
                path_weights = torch.matmul(w.view(module.out_channels, -1), path_weights)
            elif isinstance(module, torch.nn.Linear):
                norm = torch.linalg.norm(module.weight)
                w = module.weight / norm
                path_weights = torch.matmul(w, path_weights)
            
            if name == layer_b:
                total_effects += path_weights[channel_num]
                path_counts += (path_weights[channel_num] != 0).float()
                current_layer = None
    
    total_effects /= path_counts
    
    channel_indices = torch.arange(num_channels_a)
    strongest_channels = torch.stack((channel_indices, total_effects), dim=1)
    
    strongest_channels = strongest_channels[strongest_channels[:, 1].argsort(descending=True)]
    
    # Convert tensor to list of tuples
    strongest_channels = strongest_channels.cpu().numpy()
    strongest_channels = [(int(idx), float(effect)) for idx, effect in strongest_channels]
    
    return strongest_channels

if __name__ == '__main__':
    checkpoint_dirs_path = 'feature_images/my_model'
    layer_name = 'layer2.0.conv0'
    channel_num = 42

    plot_info_over_training(checkpoint_dirs_path, layer_name, channel_num, 'Activation')
    plot_feature_images_over_training(checkpoint_dirs_path, layer_name, channel_num)

# For debugging
if __name__ == "__main__":
    model = load_torchvision_model('inception_v3')
    visualize_features(model, layer_names=['AuxLogits.conv0.conv'])
