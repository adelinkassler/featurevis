import os
from vision import longact
import torch

def test_get_longitudinal_activations():
    model_name = 'resnet18'
    checkpoint_paths = ['tests/test_data/test_model.pth']
    layer_name = 'layer1.0.conv1'
    channel_num = 0
    feature_image = torch.rand(1,3,224,224)

    activations = longact.get_longitudinal_activations(model_name, checkpoint_paths, layer_name, channel_num, feature_image)
    
    assert isinstance(activations, dict)
    assert len(activations) == 1
    assert isinstance(list(activations.values())[0], float)

def test_plot_longitudinal_activations(tmpdir):
    activations = {'checkpoint_0': 0.5, 'checkpoint_1': 0.6}
    layer_name = 'layer1.0.conv1'
    channel_num = 0
    output_path = f"{tmpdir}/test_longact_plot.png"

    longact.plot_longitudinal_activations(activations, layer_name, channel_num, output_path)

    assert os.path.exists(output_path)