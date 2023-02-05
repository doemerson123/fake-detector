import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import models
from sklearn.metrics import confusion_matrix
from utils.data_pipeline_utils import file_directory


def save_confusion_matrix(
    model: Sequential,
    model_name: str,
    test_dataset: tf.constant,
    test_f1: float,
    artifacts_dir: str,
) -> None:
    """
    Saves confusion matrix as jpg with a filname using model_name and F1 score
    """

    predictions = model.predict(test_dataset)
    y_hat = np.argmax(predictions, axis=1)
    y_true = np.concatenate([y for x, y in test_dataset], axis=0)
    y_true = np.argmax(y_true, axis=1)

    confusion_matrix_obj = confusion_matrix(y_true, y_hat)

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(
        confusion_matrix_obj,
        annot=True,
        fmt="d",
        linewidths=0.75,
        cbar=False,
        ax=ax,
        cmap="Blues",
        linecolor="white",
    )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    file_name = f"{model_name}_{test_f1}_confusion_matrix.jpg"

    plt.savefig(os.path.join(artifacts_dir, file_name))


def save_loss_curve(
    model_name: str,
    hist: dict,
    test_loss: float,
    test_acc: float,
    test_f1: float,
    artifacts_dir: str,
) -> None:
    """
    Saves loss curve as jpg with a filname using model_name and performance 
    metrics. 
    """

    loss = hist.history["loss"]
    accuracy = hist.history["accuracy"]
    val_loss = hist.history["val_loss"]
    val_accuracy = hist.history["val_accuracy"]

    # number of epochs completed = len of any hist metric
    epochs = len(loss)

    plt.figure(figsize=(16, 4))
    for i, metrics in enumerate(
        zip([loss, accuracy], [val_loss, val_accuracy], ["Loss", "Accuracy"])
    ):
        plt.subplot(1, 2, i + 1)
        plt.plot(range(epochs), metrics[0], label=f"Training {metrics[2]}")
        plt.plot(range(epochs), metrics[1], label=f"Validation {metrics[2]}")
        plt.legend()

    file_name = f"{model_name}_{test_loss}_{test_acc}_{test_f1}_loss_curve.jpg"

    plt.savefig(os.path.join(artifacts_dir, file_name))


def rounded_evaluate_metrics(
    model: Sequential, test_dataset: tf.constant, precision: int
) -> tuple():
    """
    Performes model.evaluate and returns loss, accuracy, and f1 rounded to
    the precision value parameter
    """

    test_loss, test_accuracy, test_f1 = model.evaluate(test_dataset)
    rounded_loss = round(test_loss, precision)
    rounded_accuracy = round(test_accuracy, precision)
    rounded_f1 = round(test_f1, precision)

    return rounded_loss, rounded_accuracy, rounded_f1


def multiple_save_model(model: Sequential, model_name: str, path: str):
    """
    Saves trained models as hdf5 and h5. However these backups will not be the 
    best performant model that is saved as a callback using the .pb format. 

    Alternative file formats are helpful when hosting for inference
    """

    # create new directory for model files

    saved_models_dir = os.path.join(path, f"{model_name} h5 Files")
    os.makedirs(saved_models_dir)

    model.save(os.path.join(saved_models_dir, f"{model_name}.hdf5"))
    model.save(os.path.join(saved_models_dir, f"{model_name}.h5"))


def save_performance_artifacts(
    model: Sequential, model_name: str, hist: dict, test_dataset: tf.constant
) -> None:

    """
    Saves confusion matrix and loss curve to a artifacts directory

    Artifacts folder is renamed from model_name to include evaluation metrics ccon
    leading with test accuracy to promote easy sorting.

    From: model_name to {test_accuracy}_{model_name}_{test_loss}_{test_f1}
    """

    test_loss, test_accuracy, test_f1 = rounded_evaluate_metrics(model, test_dataset, 3)

    artifacts_dir_old = os.path.join(file_directory("artifact"), model_name)
    multiple_save_model(model, model_name, artifacts_dir_old)
    model_description = (
        f"{test_accuracy}-Acc {test_f1}-F1 {test_loss}-Loss -- {model_name}"
    )
    artifacts_dir_new = os.path.join(file_directory("artifact"), model_description)

    # if not os.path.isdir(artifacts_dir_new):
    #    os.makedirs(artifacts_dir_new)

    save_confusion_matrix(model, model_name, test_dataset, test_f1, artifacts_dir_old)
    save_loss_curve(
        model_name, hist, test_loss, test_accuracy, test_f1, artifacts_dir_old
    )
    os.rename(artifacts_dir_old, artifacts_dir_new)
