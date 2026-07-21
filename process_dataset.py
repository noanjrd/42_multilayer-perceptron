import pandas as pd

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

def normalize_data(data: pd.DataFrame):
    for col in data.columns:
        if col in ['id', 'diagnosis' ]:
            continue
        min = data[col].min()
        max = data[col].max()
        data[col] = (data[col] - min) / (max - min)

def create_prediction_dataset(dataset):
    # print(dataset)
    normalize_data(dataset)
    prediction_dataset = dataset.drop(['diagnosis'], axis=1)
    prediction_dataset.to_csv("prediction_dataset.csv")
    return

def create_training_dataset(dataset):
    normalize_data(dataset)
    dataset.to_csv("training_dataset.csv")
    return

def main():
    dataset = pd.read_csv("data.csv")
    dataset.columns = columns
    # print(dataset)
    create_prediction_dataset(dataset.copy())
    create_training_dataset(dataset.copy())
    return


if __name__ == "__main__":
    main()