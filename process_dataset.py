import pandas as pd
import math


columns = [
    "id",
    "diagnosis",
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]


def normalize_dataset(data: pd.DataFrame) -> None:
    """Min-max normalize feature columns in place, preserving diagnosis labels."""
    for col in data.columns:
        if col in ['diagnosis']:
            continue
        min = data[col].min()
        max = data[col].max()
        data[col] = (data[col] - min) / (max - min)


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
    data_80 = math.floor(len(dataset) * 0.8)
    x_train = dataset[0:data_80]
    x_prediction = dataset[data_80:len(dataset)]
    return x_train, x_prediction


def main() -> None:
    dataset = pd.read_csv("data.csv", header=None)
    dataset.columns = columns
    dataset = dataset.drop(['id'], axis=1)
    normalize_dataset(dataset)
    x_train, x_validation = split_dataset(dataset)

    create_validation_dataset(x_validation)
    create_training_dataset(x_train)


if __name__ == "__main__":
    main()
