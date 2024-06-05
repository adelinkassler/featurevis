from unittest.mock import patch
import os
import torch
from vision import visualize_features, featurevis, longact, influence

def test_full_pipeline(tmpdir):
    # Train and save a small test model
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 32, 3),
        torch.nn.ReLU()
    )
    torch.save(model.state_dict(), 'tests/test_data/test_model.pth')

    # Generate feature visualizations
    output_dir = f"{tmpdir}/test_feature_vis"
    visualize_features.visualize_features(model, output_path=output_dir, layer_names=['0', '2'], channels=[0,1], max_iterations=5)

    # Run longitudinal activation analysis
    activations = longact.get_longitudinal_activations('test_model', ['tests/test_data/test_model.pth'], '0', 0, torch.rand(1,3,224,224))
    longact.plot_longitudinal_activations(activations, '0', 0, f"{tmpdir}/test_longact_plot.png")

    # Run influence analysis
    influence.plot_saved_feature_images(output_dir, ['0', '2'], [0,1], [None], 'average', output_path=f"{tmpdir}/test_influence_plot.png")

    # Check all expected files were generated
    assert os.path.exists(f"{output_dir}/layer_0_channel_0_average.jpg")
    assert os.path.exists(f"{output_dir}/layer_0_channel_0_average.info.json")
    assert os.path.exists(f"{tmpdir}/test_longact_plot.png")
    assert os.path.exists(f"{tmpdir}/test_influence_plot.png")