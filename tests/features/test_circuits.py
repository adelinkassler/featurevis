from vision import circuits
import os
import torch
from unittest.mock import patch

def test_find_influential_neurons():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 32, 3),
        torch.nn.ReLU()
    )
    target_layer = '2'
    target_neuron = 0
    prev_layer = '0'
    k = 3
    influential_neurons = circuits.find_influential_neurons(model, target_layer, target_neuron, prev_layer, k)

    assert len(influential_neurons) == 3
    assert all(isinstance(i, int) for i, s in influential_neurons)
    assert all(isinstance(s, float) for i, s in influential_neurons)

@patch('vision.featurevis.load_feature_image')
def test_plot_saved_feature_images(mock_load_img, tmpdir):
    mock_load_img.return_value = torch.rand(224, 224, 3)
    images_dir = 'tests/test_data/test_feature_images'
    layer_names = ['layer1.0.conv1', 'layer1.1.conv2'] 
    channels = [0,1]
    neurons = [None]
    aggregation = 'average'
    output_path = f"{tmpdir}/test_influence_plot.png"

    circuits.plot_saved_feature_images(images_dir, layer_names, channels, neurons, aggregation, output_path=output_path)

    # Check plot file was generated
    assert os.path.exists(output_path)