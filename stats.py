from TrainingData import TrainingData
import matplotlib.pyplot as plt
import numpy as np


def compute_stats(training_data: TrainingData,  softmax_prob_train, softmax_prob_valid):
    compute_stats_loss(training_data, softmax_prob_train, softmax_prob_valid)
    compute_stats_accuracy(training_data, softmax_prob_train, softmax_prob_valid)
    compute_recall_score(training_data, softmax_prob_train, softmax_prob_valid)
    print(f"epoch {len(training_data.epoch_loss)}/{training_data.args.epochs} - "
          f"loss: {training_data.epoch_loss[-1][0]:.4f} - val_loss : {training_data.epoch_loss[-1][1]:.4f}")


def compute_recall_score(training_data: TrainingData, softmax_prob_train, softmax_prob_valid):
    y_train = training_data.training_dataset['diagnosis']
    y_valid = training_data.validation_dataset['diagnosis']

    for i, diagnosis in enumerate(['M', 'B']):
        true_positive_train = np.sum((y_train == diagnosis) & (softmax_prob_train[:,i] > 0.5))
        false_negative_train = np.sum((y_train == diagnosis) & (softmax_prob_train[:,i] < 0.5))
        recall_train = (true_positive_train) / (true_positive_train + false_negative_train)

        true_positive_valid = np.sum((y_valid == diagnosis) & (softmax_prob_valid[:,i] > 0.5))
        false_negative_valid = np.sum((y_valid == diagnosis) & (softmax_prob_valid[:,i] < 0.5))
        recall_valid = (true_positive_valid) / (true_positive_valid + false_negative_valid)

        if diagnosis == 'M':
            training_data.epoch_recall_malignant.append((recall_train, recall_valid))
        else:
            training_data.epoch_recall_benign.append((recall_train, recall_valid))


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
    patience = training_data.args.patience

    y_train = (training_dataset['diagnosis'] == 'M').astype(int).to_numpy()
    y_valid = (validation_dataset['diagnosis'] == 'M').astype(int).to_numpy()

    p_train = softmax_prob_train
    p_valid = softmax_prob_valid
    loss_train = -np.mean(y_train * np.log(p_train[:, 0]) + (1 - y_train) * np.log(p_train[:, 1]))
    loss_valid = -np.mean(y_valid * np.log(p_valid[:, 0]) + (1 - y_valid) * np.log(p_valid[:, 1]))

    if training_data.best_loss is None:
        training_data.best_loss = loss_valid
    else:
        training_data.best_loss = min(training_data.best_loss, loss_valid)
    if len(epoch_loss) >= patience:
        recent_validation_losses = np.array(epoch_loss[-patience:])[:, 1]
        training_data.early_stop = np.all(recent_validation_losses >= training_data.best_loss)

    epoch_loss.append((loss_train, loss_valid))


def display_F1(training_data: TrainingData):
    f1_maligant = 2 * (np.array(training_data.epoch_recall_malignant) * np.array(training_data.epoch_accuracy)) / (np.array(training_data.epoch_recall_malignant) + np.array(training_data.epoch_accuracy))
    f1_benign = 2 * (np.array(training_data.epoch_recall_benign) * np.array(training_data.epoch_accuracy)) / (np.array(training_data.epoch_recall_benign) + np.array(training_data.epoch_accuracy))

    plt.plot(f1_maligant[:,0], label="Malignant training F1")
    plt.plot(f1_maligant[:,1], label="Malignant validation F1")
    plt.plot(f1_benign[:,0], label="Benign training F1")
    plt.plot(f1_benign[:,1], label="Benign validation F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("F1")
    plt.legend()
    plt.show()


def display_accuracy(training_data: TrainingData):
    recall_maligant = np.array(training_data.epoch_recall_malignant)
    recall_benign = np.array(training_data.epoch_recall_benign)
    plt.plot(recall_maligant[:,0], label="Malignant training recall")
    plt.plot(recall_maligant[:,1], label="Malignant validation recall")
    plt.plot(recall_benign[:,0], label="Benign training loss")
    plt.plot(recall_benign[:,1], label="Benign validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Recall")
    plt.legend()
    plt.show()

def display_recall(training_data: TrainingData):
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
    print(training_data.epoch_accuracy[-1])
    display_F1(training_data)
    display_loss(training_data)
    display_accuracy(training_data)
    display_recall(training_data)
