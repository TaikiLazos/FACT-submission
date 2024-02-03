import time
import os
import argparse
import json
import numpy as np
import torch
import random

from torch.utils.tensorboard import SummaryWriter


def save_results(results, args):
    # Generate a unique identifier, e.g., a timestamp
    unique_id = time.strftime("%Y%m%d%H%M")

    # Create a dictionary with both results and arguments
    data = {'results': results, 'arguments': vars(args)}

    # Convert the dictionary to JSON format
    json_data = json.dumps(data, indent=4)

    arg_str = "_".join([f"{key}={value}" for key, value in vars(args).items()])

    # Define the filename using the unique identifier and arguments
    filename = f"results/results_{unique_id}_{arg_str}.json"

    # Save the JSON data to a file
    with open(filename, 'w', encoding="utf-8") as file:
        file.write(json_data)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Grid Search Script")
    parser.add_argument("--alphas", nargs="+", type=float, default=[1.0],
                        help="List of alpha values")
    parser.add_argument("--betas", nargs="+", type=float, default=[1.0],
                        help="List of alpha values")
    parser.add_argument("--gammas", nargs="+", type=float, default=[1.0],
                        help="List of gamma values")
    parser.add_argument("--lams", nargs="+", type=float, default=[1.0],
                        help="List of lambda values")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of epochs")
    parser.add_argument("--test_epochs", type=int, default=500,
                        help="Number of test epochs")
    parser.add_argument("--dataset", type=str, default="pokec_z",
                        choices=['pokec_z', 'pokec_n', 'nba', 'citeseer', 'cora', 'pubmed'],
                        help="Dataset name")
    parser.add_argument("--model_lr", nargs="+", type=float, default=[1e-4],
                        help="model learning rate")
    parser.add_argument("--lr", nargs="+", type=float, default=[1e-3],
                        help="classifier learning rate")
    parser.add_argument("--wd", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--hidden", nargs="+", type=int, default=[128],
                        help="Classifier hidden size")
    parser.add_argument("--lp", action="store_true", default=False,
                        help="Specify whether to use link prediction variant or not")
    parser.add_argument("--result_path", type=str, default='results',
                        help="Specify the path to save your results")
    parser.add_argument("--state", type=str, default="normal",
                        help="Specify the version 'ep' or 'fm'")
    args = parser.parse_args()

    # Filter out default arguments
    adjusted_args = {k: v for k, v in vars(args).items() if v != parser.get_default(k)}

    return args, adjusted_args


def set_seed(seed=20):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_writer(
    experiment_name: str, model_name: str, epochs, hparams
) -> SummaryWriter:
    """
    Create a SummaryWriter object for logging the training and test results.

    Args:
        experiment_name (str): The name of the experiment.
        model_name (str): The name of the model.

    Returns:
        SummaryWriter: The SummaryWriter object.
    """

    timestamp = time.strftime("%Y%m%d%H%M")
    hyperparams_str = '_'.join([f"{key}_{value}" for key, value in hparams.items()])
    
    # Construct the log directory path
    log_dir = os.path.join(
        "results/tensorboard",
        experiment_name,
        model_name,
        f"{epochs}",
        hyperparams_str,
        timestamp,
    ).replace("\\", "/")
    
    return SummaryWriter(log_dir=log_dir)