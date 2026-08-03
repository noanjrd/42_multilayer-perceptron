import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="?", # how many values '?' for 0 or 1, '+' for one or more
        required=False,
        default=100,
        help="Number of training epochs"
        )
    
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        required=False,
        default=[5,4],
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

    return parser.parse_args()