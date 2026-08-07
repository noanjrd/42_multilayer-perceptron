import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="?", # how many values '?' for 0 or 1, '+' for one or more
        required=False,
        default=50,
        help="Number of training epochs"
        )

    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+", # how many values '?' for 0 or 1, '+' for one or more
        required=False,
        default=["training_dataset.csv", "validation_dataset.csv"],
        help="Name of the training and validation datasets"
        )
    
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        required=False,
        default=[2,1],
        help="Number of hidden layers and their neurons"
        )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        required=False,
        default=10,
        help="Size of the batches"
        )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        nargs="?",
        required=False,
        default=1,
        help="Learning rate value"
        )

    parser.add_argument(
        "--patience",
        type=int,
        nargs="?",
        required=False,
        default=15,
        help=""
        )

    return parser.parse_args()