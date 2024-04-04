from torchvision.models import resnet18
from vision.featurevis import *

model = resnet18(pretrained=True)

# Set parameters for activation maximization
actmax_params = dict(
    # Optimizer and convergence parameters
    max_iterations=10000,
    min_iterations=5,
    convergence_threshold=1e-4,
    convergence_window=5,
    lr_schedule='exponential',
    learning_rate=0.1,
    lr_decay_rate=0.999,
    lr_min=None,
    lr_warmup_steps=0,
    # Regularization parameters
    reg_lambda=0.01,
    # use_jitter=True,
    # jitter_scale=0.05,
    # use_gauss=True,
    # gauss_kernel_size=3,
    # use_tv_reg=True,
    # tv_weight=1e-6,
    # Image parameters
    feature_image_size=224,
    crop_factor=2
)

# Run a visualize features with the test values
test_features = visualize_features(
    model,
    # layer_names=[name for name, _ in model.named_modules() if name.startswith('layer4')],
    # layer_names=['layer4', 'layer4.1', 'layer4.1.conv2'],
    layer_names=['layer1', 'layer2', 'layer3', 'layer4'],
    channels = None,
    neurons = None,
    aggregation='average',
    **actmax_params,
    batch_size=10,
    # checkpoint_path='/content/feature_images/tmp_baseline',
    output_path='../feature_images/resnet18/baseline_pt'
)
