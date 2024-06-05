import sys, os
# sys.path.append(f"{os.getenv('HOME')}/vision/")
from vision.featurevis import *
import argparse
import pdb
import subprocess

def parse_comma_separated_ints(string):
    return [int(x.strip()) for x in string.split(',')]

def parse_comma_separated_int_tuples(string):
    return [tuple(int(y) for y in x.strip().split(',')) for x in string.split(';')]

def parse_comma_separated_strs(string):
    return [x.strip() for x in string.split(',')]

def is_slurm_available():
    try:
        result = subprocess.run(['sinfo', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("Slurm is available on this machine.")
            return True
        else:
            print("Slurm is not available on this machine.")
            return False
    except FileNotFoundError:
        print("Slurm is not available on this machine.")
        return False


def main():
    """
    Entry point of the script, parses command-line arguments and calls `visualize_features` with appropriate parameters.

    This function sets up the command-line interface for generating feature visualizations. It parses the provided arguments,
    loads the specified model, and calls the `visualize_features` function from `featurevis.py` with the appropriate parameters.
    The generated feature visualizations are saved to the specified output directory.
    """

    parser = argparse.ArgumentParser(description='Visualize features using activation maximization')
    parser.add_argument('--model', type=str, help='Model name (must match class name in torchvision)')
    parser.add_argument('--checkpoint-path', type=str, nargs='+', help='Path to saved pytorch model')
    parser.add_argument('--layer-names', '--layers', type=str, nargs='+', help='Whitespace-separated list of layer names (defaults to all convolational layers)')
    parser.add_argument('--channels', type=int, nargs='+', help='Whitespace-separated list of channels to visualize (defaults to all channels)')
    parser.add_argument('--neurons', type=parse_comma_separated_int_tuples, help='Semicolon-separated list of comma-separated tuples of neurons to visualize')
    parser.add_argument('--aggregation', type=str, default='average', help='Aggregation method')
    parser.add_argument('--number-of-images', type=int, default=1, help="Number of feature images to produce (penalized by diversity weight)")
    parser.add_argument('--crop-factor', type=int, default=2, help='Crop factor')
    parser.add_argument('--init-images-dir', type=str, nargs='+', help='Path to directory of feature images to load as starting point (or a list of such paths, for fallback dirs)')
    parser.add_argument('--output-path', type=str, help='Output path')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU')
    parser.add_argument('--parallel', action='store_true', help='Run in parallel using SLURM')
    parser.add_argument('--local', action='store_true', help="Force local running even if SLURM is available")
    parser.add_argument('--seed', type=int, help='Set seed for activation maximization function')
    parser.add_argument('--config', '-c', type=str, help="Path to a yaml-formatted config file with parameters. Flags will override.")

    # Optimizer and convergence parameters
    parser.add_argument('--max-iterations', type=int, default=1000, help='Maximum number of iterations')
    parser.add_argument('--min-iterations', type=int, default=2, help='Minimum number of iterations')
    parser.add_argument('--convergence-threshold', type=float, default=1e-4, help='Stop iterating when the loss changes by less than this amount')
    parser.add_argument('--convergence-window', type=int, default=2, help='Number of iterations over which to check the convergence threshold')
    parser.add_argument('--lr-schedule', type=str, default=None, help="Learning rate schedule: supports 'exponential', 'cosine', leave unset for fixed rate")
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--lr-decay-rate', type=float, default=None, help='Learning rate decay rate in exponential schedule')
    parser.add_argument('--lr-min', type=float, default=None, help='Minimum learning rate for cosine annealing schedule')
    parser.add_argument('--lr-warmup-steps', type=int, default=0, help='Scale up linearly from lr_min to learning_rate over this many steps')

    # Regularization parameters #TODO: *definitely* rewrite these help strings
    parser.add_argument('--reg-lambda', type=float, default=0.01, help='Regularization lambda')
    parser.add_argument('--use-jitter', action='store_true', help='Use jitter')
    parser.add_argument('--jitter-scale', type=float, default=0.05, help='Jitter scale')
    parser.add_argument('--use-scaling', action='store_true', help='Use scaling')
    parser.add_argument('--scale-range', type=float, default=0.1, help='Scale range')
    parser.add_argument('--use-gauss', action='store_true', help='Use Gaussian regularization')
    parser.add_argument('--gauss-kernel-size', type=int, default=3, help='Gaussian kernel size')
    parser.add_argument('--use-tv-reg', action='store_true', help='Use total variation regularization')
    parser.add_argument('--tv-weight', type=float, default=1e-6, help='Total variation regularization weight')
    parser.add_argument('--use-decorrelation', action='store_true', help='Use decorrelation regularization (NOT SUPPORTED)')
    parser.add_argument('--decorrelation-weight', type=float, default=0.1, help='Decorrelation regularization weight')
    parser.add_argument('--diversity-weight', type=float, default=1.0, help="Weight to use for cosine similarity penalty between feature images")

    # Image parameters # TODO: double check
    parser.add_argument('--input-image', type=str, default=None, help='NOT SUPPORTED')
    parser.add_argument('--feature-image-size', type=int, default=224, help='Feature image size')
    # parser.add_argument('--crop-factor', type=int, default=2, help='Crop factor')

    # Misc arguments
    # parser.add_argument('--use-gpu', action='store_true', help='Use GPU')
    parser.add_argument('--progress-bar', action='store_true', help='Show progress bar')
    args = parser.parse_args()

    # if args.config:
    #     if not os.path.isfile(args.config):
            

    if args.output_path is None:
        if input("--output-path has not been set. Continue without saving results? y/[n] ") != 'y':
            exit()
    
    # Create a loader function for initial images
    init_image_loader = make_feature_image_loader_with_fallback(args.init_images_dir) if \
        args.init_images_dir is not None else None

    if args.parallel and args.local:
        raise ValueError("Cannot set both --parallel and --local flags")
    elif args.parallel:
        parallel = True
    elif args.local:
        parallel = False
    else:
        parallel = is_slurm_available()
        print("Determining parallelization automatically since neither --parallel nor --local were set. " + 
              f"Result: {'parallel' if parallel else 'local'} based on automatically detected slurm availability")

    act_max_params = {
        'max_iterations': args.max_iterations,
        'min_iterations': args.min_iterations,
        'convergence_threshold': args.convergence_threshold,
        'convergence_window': args.convergence_window,
        'lr_schedule': args.lr_schedule,
        'learning_rate': args.learning_rate,
        'lr_decay_rate': args.lr_decay_rate,
        'lr_min': args.lr_min,
        'lr_warmup_steps': args.lr_warmup_steps,
        'reg_lambda': args.reg_lambda,
        'use_jitter': args.use_jitter,
        'jitter_scale': args.jitter_scale,
        'use_scaling': args.use_scaling,
        'scale_range': args.scale_range,
        'use_gauss': args.use_gauss,
        'gauss_kernel_size': args.gauss_kernel_size,
        'use_tv_reg': args.use_tv_reg,
        'tv_weight': args.tv_weight,
        'use_decorrelation': args.use_decorrelation,
        'decorrelation_weight': args.decorrelation_weight,
        # 'input_image': args.input_image,
        'feature_image_size': args.feature_image_size,
        # 'crop_factor': args.crop_factor,
        # 'use_gpu': args.use_gpu,
        'progress_bar': args.progress_bar,
        'seed': args.seed
    }

    job_arrays = []
    if args.checkpoint_path:
        checkpoint_paths = args.checkpoint_path
        for checkpoint_path in checkpoint_paths:
            try:
                output_path = os.path.join(args.output_path, os.path.basename(checkpoint_path))
                model = load_torchvision_model(args.model, checkpoint_path)
                job_array = visualize_features(model, layer_names=args.layer_names, channels=args.channels, neurons=args.neurons,
                                            aggregation=args.aggregation, number_of_images=args.number_of_images, crop_factor=args.crop_factor,
                                            init_image_loader=init_image_loader, output_path=output_path,
                                            batch_size=args.batch_size, use_gpu=args.use_gpu, parallel=parallel, return_output=False,
                                            **act_max_params)
                job_arrays.append(job_array)
            except:
                print(f"Exception reached for {checkpoint_path}")
    else:
        model = load_torchvision_model(args.model)
        job_array = visualize_features(model, layer_names=args.layer_names, channels=args.channels, neurons=args.neurons,
                                        aggregation=args.aggregation, number_of_images=args.number_of_images, crop_factor=args.crop_factor,
                                        init_image_loader=init_image_loader, output_path=args.output_path,
                                        batch_size=args.batch_size, use_gpu=args.use_gpu, parallel=parallel, return_output=False,
                                        **act_max_params)
        job_arrays.append(job_array)
    
    if args.parallel:
        track_job_array_progress(job_arrays)

if __name__ == '__main__':
    main()