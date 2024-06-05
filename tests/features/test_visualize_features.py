import os
from unittest.mock import patch
from vision import visualize_features
import torch

@patch('vision.featurevis.activation_maximization')
def test_visualize_features(mock_act_max):
    # Set up mock return values 
    mock_act_max.return_value = (torch.rand(1,3,224,224), 0.5, [0.1,0.2,0.3], 3)
    
    model = torch.nn.Sequential(torch.nn.Conv2d(3,16,3))
    output_dir = 'tests/test_data/test_feature_vis_cli'

    visualize_features.visualize_features(model, output_path=output_dir, layer_names=['0'], channels=[0,1], max_iterations=1)

    # Check activation_maximization was called correctly
    mock_act_max.assert_any_call(model, layer_name='0', channel=0, max_iterations=1)
    mock_act_max.assert_any_call(model, layer_name='0', channel=1, max_iterations=1)

    # Check files were generated
    assert os.path.exists(f"{output_dir}/layer_0_channel_0_average.jpg")
    assert os.path.exists(f"{output_dir}/layer_0_channel_0_average.info.json")
    assert os.path.exists(f"{output_dir}/layer_0_channel_1_average.jpg")
    assert os.path.exists(f"{output_dir}/layer_0_channel_1_average.info.json")