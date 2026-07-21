import pandas as pd
import argparse
from Layer import Layer

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
        type=int,
        nargs="?",
        required=False,
        default=1,
        help="Learning rate value"
        )
    # print(parser)
    return parser.parse_args()

def forward_propagation(layers, data):
    for layer in layers:
        print()

def start_training(args):
    layers = [Layer(args.layers[i], args.layers[i-1]) for i in range(1,len(args.layers))]
    layers = [Layer(args.layers[0], 13)] + layers
    print(len(layers))
    data = pd.read_csv('training_dataset.csv')
    forward_propagation(layers, data)
    return

def main():
    args = parse_args()
    layers = args.layers
    # parse_args()
    print(layers)
    start_training(args)
    return

if __name__ == "__main__":
    main()