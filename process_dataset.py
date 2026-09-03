import pandas as pd
import math


def create_validation_dataset(dataset: pd.DataFrame) -> None:
    """Write the validation split to ``validation_dataset.csv``."""
    dataset.to_csv("validation_dataset.csv", index=False)
    return


def create_training_dataset(dataset: pd.DataFrame) -> None:
    """Write the training split to ``training_dataset.csv``."""
    dataset.to_csv("training_dataset.csv", index=False)
    return


def split_dataset(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle a dataset and return random 80/20 training and validation splits."""
    dataset = dataset.sample(n=len(dataset))
    data_80 = math.floor(len(dataset) * 0.80)
    x_train = dataset[0:data_80]
    x_prediction = dataset[data_80:len(dataset)]
    return x_train, x_prediction


def main() -> None:
    dataset = pd.read_csv("data.csv", header=None)
    x_train, x_validation = split_dataset(dataset)

    create_validation_dataset(x_validation)
    create_training_dataset(x_train)


if __name__ == "__main__":
    main()
