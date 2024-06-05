import os
import torch
from vision import featurevis

def test_load_torchvision_model():
    model = featurevis.load_torchvision_model('resnet18')
    assert isinstance(model, torch.nn.Module)

    model = featurevis.load_torchvision_model('resnet18', 'tests/test_data/test_model.pth') 
    assert isinstance(model, torch.nn.Module)

def test_preprocess_image():
    img = torch.rand(3, 224, 224)
    preprocessed = featurevis.preprocess_image(img)
    assert preprocessed.shape == (1, 3, 224, 224) 
    assert preprocessed.min() >= 0
    assert preprocessed.max() <= 1

def test_deprocess_image():
    img = torch.rand(1, 3, 224, 224)
    deprocessed = featurevis.deprocess_image(img)
    assert deprocessed.shape == (224, 224, 3)
    assert deprocessed.min() >= 0
    assert deprocessed.max() <= 255
    assert deprocessed.dtype == torch.uint8

def test_activation_maximization():
    model = featurevis.load_torchvision_model('resnet18')
    layer_name = 'layer1.0.conv1'
    channel = 0
    img, act, losses, i = featurevis.activation_maximization(model, layer_name, channel, max_iterations=5)
    assert isinstance(img, torch.Tensor)
    assert isinstance(act, float)
    assert isinstance(losses, list)
    assert isinstance(i, int)