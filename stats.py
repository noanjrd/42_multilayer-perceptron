from TrainingData import TrainingData
import matplotlib.pyplot as plt
import numpy as np
import copy
import pandas as pd


def compute_stats(training_data: TrainingData,  softmax_prob_train: np.ndarray, softmax_prob_valid: np.ndarray) -> None:
    """Compute and record all training and validation metrics for one epoch."""
    compute_stats_loss(training_data, softmax_prob_train, softmax_prob_valid)
    compute_stats_accuracy(training_data, softmax_prob_train, softmax_prob_valid)
    compute_stats_precision(training_data, softmax_prob_valid)
    compute_recall_score(training_data, softmax_prob_train, softmax_prob_valid)
    print(f"epoch {len(training_data.epoch_loss)}/{training_data.args.epochs[0]} - "
          f"loss: {training_data.epoch_loss[-1][0]:.4f} - val_loss : {training_data.epoch_loss[-1][1]:.4f}")


def compute_recall_score(training_data: TrainingData, softmax_prob_train: np.ndarray, softmax_prob_valid: np.ndarray) -> None:
    """Record per-class recall for the current training and validation outputs."""
    y_train = training_data.training_dataset['diagnosis']
    y_valid = training_data.validation_dataset['diagnosis']

    for i, diagnosis in enumerate(['M', 'B']):
        true_positive_train = np.sum((y_train == diagnosis) & (softmax_prob_train[:, i] > 0.5))
        false_negative_train = np.sum((y_train == diagnosis) & (softmax_prob_train[:, i] < 0.5))
        recall_train = (true_positive_train) / (true_positive_train + false_negative_train)

        true_positive_valid = np.sum((y_valid == diagnosis) & (softmax_prob_valid[:, i] > 0.5))
        false_negative_valid = np.sum((y_valid == diagnosis) & (softmax_prob_valid[:, i] < 0.5))
        recall_valid = (true_positive_valid) / (true_positive_valid + false_negative_valid)

        if diagnosis == 'M':
            training_data.epoch_recall_malignant.append((recall_train, recall_valid))
        else:
            training_data.epoch_recall_benign.append((recall_train, recall_valid))


def compute_stats_accuracy(training_data: TrainingData,  softmax_prob_train: np.ndarray, softmax_prob_valid: np.ndarray) -> None:
    """Record classification accuracy for the training and validation datasets."""
    prediction_train = np.where(softmax_prob_train[:, 0] > softmax_prob_train[:, 1], 'M', 'B')
    prediction_valid = np.where(softmax_prob_valid[:, 0] > softmax_prob_valid[:, 1], 'M', 'B')
    prediction_train = np.where(training_data.training_dataset['diagnosis'] == prediction_train, True, False)
    prediction_valid = np.where(training_data.validation_dataset['diagnosis'] == prediction_valid, True, False)
    score_train = prediction_train.sum() / len(training_data.training_dataset)
    score_valid = prediction_valid.sum() / len(training_data.validation_dataset)
    # print(score_valid)
    # score_train = score_train.merge(score_valid)
    training_data.epoch_accuracy.append((score_train, score_valid))
    return


def compute_stats_precision(training_data: TrainingData, softmax_prob_valid: np.ndarray) -> None:
    """Record validation precision for the malignant and benign classes."""
    prediction_valid: np.ndarray = np.where(softmax_prob_valid[:, 0] > softmax_prob_valid[:, 1], 'M', 'B')
    m_indices_valid: np.ndarray = np.where(prediction_valid == 'M')[0]
    b_indices_valid: np.ndarray = np.where(prediction_valid == 'B')[0]
    validation_dataset_m: pd.DataFrame = training_data.validation_dataset.iloc[m_indices_valid]
    validation_dataset_b: pd.DataFrame = training_data.validation_dataset.iloc[b_indices_valid]

    percentage_m: float
    percentage_b: float
    if len(m_indices_valid) > 0:
        percentage_m = len(validation_dataset_m[validation_dataset_m['diagnosis'] == 'M']) / len(m_indices_valid)
    else:
        percentage_m = 0.0
    if len(b_indices_valid) > 0:
        percentage_b = len(validation_dataset_b[validation_dataset_b['diagnosis'] == 'B']) / len(b_indices_valid)
    else:
        percentage_b = 0.0

    training_data.epoch_precision.append((percentage_m, percentage_b))


def compute_stats_loss(training_data: TrainingData,  softmax_prob_train: np.ndarray, softmax_prob_valid: np.ndarray) -> None:
    """Record cross-entropy losses and update early-stopping state."""
    training_dataset = training_data.training_dataset
    validation_dataset = training_data.validation_dataset
    epoch_loss = training_data.epoch_loss
    patience = training_data.args.patience

    y_train = (training_dataset['diagnosis'] == 'M').astype(int).to_numpy()
    y_valid = (validation_dataset['diagnosis'] == 'M').astype(int).to_numpy()

    p_train = softmax_prob_train
    p_valid = softmax_prob_valid
    loss_train = -np.mean(y_train * np.log(p_train[:, 0]) + (1 - y_train) * np.log(p_train[:, 1]))
    loss_valid = -np.mean(y_valid * np.log(p_valid[:, 0]) + (1 - y_valid) * np.log(p_valid[:, 1]))

    epoch_loss.append((loss_train, loss_valid))

    if training_data.best_loss is None or loss_valid < training_data.best_loss:
        training_data.best_loss = loss_valid
        training_data.best_layers = copy.deepcopy(training_data.layers)

    if len(epoch_loss) >= patience:
        recent_validation_losses = np.array(epoch_loss[-patience:])[:, 1]
        training_data.early_stop = bool(np.all(recent_validation_losses > training_data.best_loss))


def display_stats(training_data: TrainingData) -> None:
    """Print the final macro F1 score and plot metric histories by epoch."""
    loss = np.array(training_data.epoch_loss)
    accuracy = np.array(training_data.epoch_accuracy)
    precision = np.array(training_data.epoch_precision)
    recall_m = np.array(training_data.epoch_recall_malignant)
    recall_b = np.array(training_data.epoch_recall_benign)

    # Validation F1 per class
    f1_m = 2 * precision[:, 0] * recall_m[:, 1] / (precision[:, 0] + recall_m[:, 1] + 1e-8)
    f1_b = 2 * precision[:, 1] * recall_b[:, 1] / (precision[:, 1] + recall_b[:, 1] + 1e-8)
    macro_f1 = (f1_m[-1] + f1_b[-1]) / 2
    print(f"Macro validation F1-score: {macro_f1:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Loss
    axes[0, 0].plot(loss[:, 0], label="Training")
    axes[0, 0].plot(loss[:, 1], label="Validation")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()

    # Accuracy
    axes[0, 1].plot(accuracy[:, 0], label="Training")
    axes[0, 1].plot(accuracy[:, 1], label="Validation")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].legend()

    # Validation precision and recall
    axes[1, 0].plot(precision[:, 0], label="Malignant precision")
    axes[1, 0].plot(precision[:, 1], label="Benign precision")
    axes[1, 0].plot(recall_m[:, 1], label="Malignant recall")
    axes[1, 0].plot(recall_b[:, 1], label="Benign recall")
    axes[1, 0].set_title("Validation precision and recall")
    axes[1, 0].legend()

    # Validation F1
    axes[1, 1].plot(f1_m, label="Malignant F1")
    axes[1, 1].plot(f1_b, label="Benign F1")
    axes[1, 1].set_title("Validation F1")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    plt.show()
