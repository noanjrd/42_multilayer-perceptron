from Layer import Layer
import pandas as pd
import argparse


class TrainingData:
    """Bundle datasets, model state, and metrics used during training."""

    def __init__(self, training_dataset: pd.DataFrame, validation_dataset: pd.DataFrame, args: argparse.Namespace, layers: list[Layer]):
        """Initialize the mutable state shared by the training pipeline."""
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.args = args
        self.layers = layers
        self.epoch_loss: list = []
        self.epoch_accuracy: list = []
        self.epoch_precision: list = []
        self.epoch_recall_malignant: list = []
        self.epoch_recall_benign: list = []
        self.best_loss: list[Layer] | None = None
        self.early_stop = False
        self.best_layers: list[Layer] | None = None
