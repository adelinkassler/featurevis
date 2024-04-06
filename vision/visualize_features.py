import argparse
from vision.featurevis import *

def main():
    parser = argparse.ArgumentParser(description='Visualize features using activation maximization')
    parser.add_argument('--model', type=str, default='resnet18', help='Model name')
    parser.add_argument('--layer-names', type=str, nargs='+', help='Layer names')
    parser.add_argument('--channels', type=int, nargs='+', help='Channels to visualize')
    parser.add_argument('--neurons', type=int, nargs='+', help='Neurons to visualize')
    parser.add_argument('--aggregation', type=str, default='average', help='Aggregation method')
    parser.add_argument('--crop-factor', type=int, default=2, help='Crop factor')
    parser.add_argument('--checkpoint-path', type=str, help='Checkpoint path')
    parser.add_argument('--output-path', type=str, help='Output path')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU')
    parser.add_argument('--parallel', action='store_true', help='Run in parallel using SLURM')
    args = parser.parse_args()

    model = load_resnet18()
    visualize_features(model, layer_names=args.layer_names, channels=args.channels, neurons=args.neurons,
                       aggregation=args.aggregation, crop_factor=args.crop_factor,
                       checkpoint_path=args.checkpoint_path, output_path=args.output_path,
                       batch_size=args.batch_size, use_gpu=args.use_gpu, parallel=args.parallel, return_output=False)

if __name__ == '__main__':
    main()