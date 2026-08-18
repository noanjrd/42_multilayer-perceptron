from Layer import Layer


class TrainingData:
    def __init__(self, training_dataset, validation_dataset, args, layers: list[Layer]):
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.args = args
        self.layers = layers
        self.epoch_loss = []
        self.epoch_accuracy = []
        self.epoch_precision = []
        self.epoch_recall_malignant = []
        self.epoch_recall_benign = []
        self.best_loss = None
        self.early_stop = False
        self.best_layers = None
