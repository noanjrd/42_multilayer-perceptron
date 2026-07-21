import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="?", # how many values '?' for 0 or 1, '+' for one or more
        required=False,
        default=1000,
        help="Number of training epochs"
        )
    
    parser.add_argument(
        "--layer",
        type=int,
        nargs="?", # how many values '?' for 0 or 1, '+' for one or more
        required=False,
        default=2,
        help="Number of hidden layers"
        )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?", # how many values '?' for 0 or 1, '+' for one or more
        required=False,
        default=10,
        help="Size of the batches"
        )
    
    parser.add_argument(
        "--learning_rate",
        type=int,
        nargs="?", # how many values '?' for 0 or 1, '+' for one or more
        required=False,
        default=1,
        help="Learning rate value"
        )
    # print(parser)
    return parser.parse_args()

def main():
    args = parse_args()
    # parse_args()
    return

if __name__ == "__main__":
    main()