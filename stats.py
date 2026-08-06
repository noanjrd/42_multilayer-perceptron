from TrainingData import TrainingData
import matplotlib.pyplot as plt
import numpy as np

def compute_stats(training_data: TrainingData,  softmax_prob_train, softmax_prob_valid):
    compute_stats_loss(training_data, softmax_prob_train, softmax_prob_valid)
    compute_stats_accuracy(training_data, softmax_prob_train, softmax_prob_valid)
    print(f"epoch {len(training_data.epoch_loss)}/{training_data.args.epochs} - loss: {training_data.epoch_loss[-1][0]:.4f} - val_loss : {training_data.epoch_loss[-1][1]:.4f}")
    return


def compute_stats_accuracy(training_data: TrainingData,  softmax_prob_train, softmax_prob_valid):
    prediction_train = np.where(softmax_prob_train[:,0] > softmax_prob_train[:,1], 'M', 'B')
    prediction_valid = np.where(softmax_prob_valid[:,0] > softmax_prob_valid[:,1], 'M', 'B')
    prediction_train = np.where(training_data.training_dataset['diagnosis'] == prediction_train, True, False)
    prediction_valid = np.where(training_data.validation_dataset['diagnosis'] == prediction_valid, True, False)
    score_train = prediction_train.sum() / len(training_data.training_dataset)
    score_valid = prediction_valid.sum() / len(training_data.validation_dataset)
    # print(score_valid)
    # score_train = score_train.merge(score_valid)
    training_data.epoch_accuracy.append((score_train, score_valid))
    return


def compute_stats_loss(training_data: TrainingData,  softmax_prob_train, softmax_prob_valid):
    training_dataset = training_data.training_dataset
    validation_dataset = training_data.validation_dataset
    epoch_loss = training_data.epoch_loss

    y_train = (training_dataset['diagnosis'] == 'M').astype(int).to_numpy()
    y_valid = (validation_dataset['diagnosis'] == 'M').astype(int).to_numpy()

    p_train = softmax_prob_train
    p_valid = softmax_prob_valid
    loss_train = -np.mean(y_train * np.log(p_train[:, 0]) + (1 - y_train) * np.log(p_train[:, 1]))
    loss_valid = -np.mean(y_valid * np.log(p_valid[:, 0]) + (1 - y_valid) * np.log(p_valid[:, 1]))
    # print(loss)
    epoch_loss.append((loss_train, loss_valid))
    return


def display_accuracy(training_data: TrainingData):
    print(training_data.epoch_accuracy[-1][1])
    stats = np.array(training_data.epoch_accuracy)
    plt.plot(stats[:,0], label="Training loss")
    plt.plot(stats[:,1], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.show()


def display_loss(training_data: TrainingData):
    stats = np.array(training_data.epoch_loss)
    plt.plot(stats[:,0], label="Training accuracy")
    plt.plot(stats[:,1], label="Validation acuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()
    plt.show()
    return

def display_stats(training_data: TrainingData):
    display_loss(training_data)
    display_accuracy(training_data)
