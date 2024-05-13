# Vision 1
Repository for Timaeus's research on vision models.


# Getting Started

Make a virtual environment and install the requirements:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

The vision.featurevis module contains various tools meant to work with visualizing features of torchvision models.

`feature_visualization.py` is a command line tool for generating large batches of feature visualizations. Run `python vision/visualize_features.py -h` for usage.