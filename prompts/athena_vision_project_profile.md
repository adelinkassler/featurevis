<project_profile>
  <goals>
    This is an AI developmental interpretability project designed to apply the tools of singular learning theory to AI training to help understand how vision models learn circuits and features during training. It follows in the footsteps of a similar project published by Timaeus research, which studied language models.

    To start, we want to visualize the features learned by a vision model (focusing on resnet to start, but will try other models like inception later). To do this, we use activation maximization to find images that maximize the activation of a given convolutional channel or neuron for a given layer of the model. Code implementing this is found in featurevis.py, and a command-line wrapper is provided by visualize_features.py. The goal is to use this code to generate features for a model at numerous checkpoints taken across the training process, and look at how and when features develop.

    We want to look at the development timeline of features in relation to information from singular learning theory (such as the local learning coefficient, which another project member is working on), and to compare features to each other. We're interested in whether we can see the model building smaller features, then bigger features out of the smaller features.
  </goals>
  The workspace is laid out like so (not all files are shown for output files, just enough to give you the idea):
  <workspace>
  ```
  ┌─ ./
  ├─ ┬ feature_images/
  │  ├─ ┬ inception_v3/
  │  ├─ ┬ resnet18/
  │  └─ ┬ resnet50/
  │     └─ ┬ checkpoints/
  │        └─ ┬ checkpoint_2024-03-25-00h28_epoch_0_train_200.pth.tar/
  │           ├─ layer_layer2_channel_0_average.info.json
  │           └─ layer_layer2_channel_0_average.jpg
  │              └─ ...
  │        └─ ┬ pretrained/
  │           ├─ ┬ friendly/
  │           └─ ┬ full/
  │              ├─ layer_layer2_channel_0_average.info.json
  │              └─ layer_layer2_channel_0_average.jpg
  │                 └─ ...
  ├─ ┬ models/
  │  └─ ┬ resnet50_checkpoints/1/
  │     ├─ checkpoint_2024-03-25-00h28_epoch_0_train_200.pth.tar
  │     └─ checkpoint_2024-03-25-00h59_epoch_0_val_200.pth.tar
  │        └─ ...
  ├─ ┬ vision/
  │  ├─ __init__.py
  │  ├─ featurevis.py
  │  ├─ visualize_features.py
  │  ├─ longact.py
  │  └─ influence.py
  ├─ setup.py
  ├─ README.md
  └─ requirements.txt
  ```
  </workspace>
  <files>
    visualize_features generates an image file and an info file for each feature. The paths are formulaic, like so:
      `feature_images/{model_name}/{sometimes_other_folders}/{checkpoint_name}/layer_{layer_name}_channel_{channel_num}_average.jpg`
      `feature_images/{model_name}/{sometimes_other_folders}/{checkpoint_name}/layer_{layer_name}_channel_{channel_num}_average.info.json`

    Model checkpoint names look like: `checkpoint_2024-03-25-00h28_epoch_0_train_200.pth.tar` and are guaranteed to appear in order within the directory because of the timestamps - we can use that to order them for longitudinal plots and analysis in our code. T
  </files>
</project_profile>

Raw:
I'm working on an AI developmental interpretability research project that uses the attached python files to generate a large number of feature images. These are images created through activation maximization with respect to a convolutional channel or neuron in the neural network (which is a vision network). Each feature image file also comes with an info file containing various parameters for that activation maximization process, including the actual amount that the image activates that particular neuron.

In particular, we're interested in how that activation and the feature image change for a particular channel over the course of training. So we've repeated this visualization process for a number of checkpoints of the same model over the course of training.

I've done a dummy pilot run with only a few checkpoints and channels, and now I'd like to test out how I can visualize the features changing over training. I'd like you to write me code that, given a folder containing all the feature images and the specification for a neuron/channel, will plot the activation over training (checkpoints are named such that they appear in training order). I also want to see a timeline of the feature images at each timepoint, so I'd like code that generates for the same neuron a grid of the individual feature images labelled with the name of their checkpoint.

The workspace is laid out like so (not all files are shown for output files, just enough to give you the idea):
<workspace>
./
﹂feature_images/
  ﹂inception_v3/
  ﹂resnet18/
  ﹂resnet50/
    ﹂checkpoints/
      ﹂checkpoint_2024-03-25-00h28_epoch_0_train_200.pth.tar/
        ﹂layer_layer2_channel_0_average.info.json
        ﹂layer_layer2_channel_0_average.jpg
      ﹂pretrained/
        ﹂friendly/
        ﹂full/
          ﹂layer_layer2_channel_0_average.info.json
          ﹂layer_layer2_channel_0_average.jpg
﹂models/
  ﹂resnet50_checkpoints/1/
    ﹂checkpoint_2024-03-25-00h28_epoch_0_train_200.pth.tar
    ﹂checkpoint_2024-03-25-00h59_epoch_0_val_200.pth.tar
﹂vision/
  ﹂__init__.py
  ﹂featurevis.py
  ﹂visualize_features.py




The file formats for the various files are structured like so:
`feature_images/{model_name}/{sometimes_other_folders}/{checkpoint_name}/layer_{layer_name}_channel_{channel_num}_average.jpg`
`feature_images/{model_name}/{sometimes_other_folders}/{checkpoint_name}/layer_{layer_name}_channel_{channel_num}_average.info.json`

checkpoint names look like: `checkpoint_2024-03-25-00h28_epoch_0_train_200.pth.tar` and are guaranteed to appear in order within the directory because of the timestamps - we can use that to order them for longitudinal plots and analysis in our code.

layer names look like: `layer2` or `conv0` or `layer2.0.conv0`. They use the standard layer names for the model as provided by torchvision.
