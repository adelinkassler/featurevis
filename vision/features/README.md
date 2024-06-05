# Features Submodule

This submodule contains code for analyzing and visualizing the features learned by AI vision models during training, as part of the larger project on developmental interpretability.

(This section of the project was developed by Adelin Kassler during her Athena research fellowship in 2024.) 

## Project Goals

The overarching goal of this project is to better understand how vision AI models develop their internal representations over the course of training, using the tools of singular learning theory. Specifically, we aim to:

1. Visualize the features learned by individual neurons and channels at different layers of the model, using activation maximization.

2. Track how these learned features change over time by comparing visualizations from different checkpoints during training.

3. Relate the trajectory of feature development to theoretical metrics from singular learning theory, such as the local learning coefficient.

4. Investigate whether the model appears to compositionally build up complex features from simpler ones learned earlier in training.

By shedding light on the learning dynamics of these models, we hope to gain insights that can help make AI systems more interpretable and inform the development of more robust and efficient learning algorithms.


## Contents

- `visualize_features.py`: The main entry point script for generating feature visualizations using activation maximization. It loads a model checkpoint, specifies which components to visualize, and runs the optimization process.

- `featurevis.py`: Contains the core activation maximization code. This includes the loss function definition, regularization techniques, and the optimization loop. The `visualize_features` function is the primary interface.

- `longact.py`: Provides functions for longitudinal analysis of how activations and feature visualizations change across training checkpoints. This includes plotting activations over time and generating comparison grids of visualizations.

- `circuits.py`: (Work in progress) Contains code for analyzing the influence connections between neurons in different layers, to start to understand the circuits that form during training.

- `utils.py`: Utility functions used across the submodule.

## Usage

To generate feature visualizations, run `visualize_features.py` with the desired model architecture, checkpoint path(s), and layer/channel/neuron specifications. For example:
```
python visualize_features.py --model resnet50 --checkpoint-path model_checkpoints/ --layer-names layer3.1 layer4 --channels 0 1 2 --output-path results/
```

This will visualize channels 0, 1, and 2 from layers 3.1 and 4 of a ResNet-50 model, using the checkpoints stored in `model_checkpoints/`, and save the results to `results/`.

Longitudinal analysis can be performed using the functions in `longact.py`. For example:

```python
from features.longact import plot_longitudinal_activations

activations = get_activations(model, layer_name, channel, checkpoint_paths)
plot_longitudinal_activations(activations, layer_name, channel, output_path)
```

This will plot the activations of a specified layer and channel across the provided checkpoint paths.  

## Parameters and Configuration

The `visualize_features.py` script provides many command-line parameters to control the visualization process, including:

- Optimization hyperparameters: learning rate, number of iterations, convergence criteria
- Regularization settings: type of regularization (L2, total variation, etc.), regularization strengths
- Output settings: number of images to generate per component, output directory

Default values are provided for all parameters. The `featurevis.py` module also contains detailed documentation on each parameter.

## Results

Visualization results are saved as image files (`.jpg` format) in the specified output directory, along with `.json` files containing metadata about the optimization process.
Longitudinal analysis plots are saved as `.png` files by default.

## Unfinished Works

The following goals of this sub-project were not fully realized by the end of the Athena program. I'd like to say that they'll be the subject of ongoing work, but realistically... "knowing how way leads on to way, I doubted that I would ever return". As such, use these modules with caution: 

- The circuits.py module, which will provide tools for analyzing the influence connections between neurons and layers. The goal is to understand how circuits and pathways form during training.
- Integration with singular learning theory metrics. We plan(ned?) to add code for computing these metrics and relating them to the visualized features.
- Extending the framework to more model architectures beyond ResNet.

Of course, if you find these tools useful for your own work, please feel free to fork and submit pull requests.

