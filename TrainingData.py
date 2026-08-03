class TrainingData:
    def __init__(self, training_dataset, validation_dataset , args, layers):
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.args = args
        self.layers = layers
        self.epoch_stats = []