
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import mean_squared_error as MSE
from utils.data_pipeline_utils import load_params
from utils.data_pipeline_util import datasets, filepath

params = load_params('fake-detector\params.yaml')
artifact_directory = filepath('artifact')

def plot_confusion_matrix(y_true, y_pred):
    confusion_matrix_obj = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(confusion_matrix_obj, 
                annot=True, 
                fmt='d', 
                linewidths=.75,  
                cbar=False, 
                ax=ax,
                cmap='Blues',
                linecolor='white')
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')


def plot_model_metrics(model_name, model, hist):
    
    loss = hist.history['loss']
    accuracy = hist.history['accuracy']
    val_loss = hist.history['val_loss']
    val_accuracy = hist.history['val_accuracy']
    
    # number of epochs completed is the len of any hist metric
    epochs = len(loss)

    plt.figure(figsize=(16, 4))
    for i, metrics in enumerate(zip([loss, accuracy], 
                                    [val_loss, val_accuracy], 
                                    ['Loss', 'Accuracy'])
                                ):
        plt.subplot(1, 2, i + 1)
        plt.plot(range(epochs), metrics[0], label=f'Training {metrics[2]}')
        plt.plot(range(epochs), metrics[1], label=f'Validation {metrics[2]}')
        plt.legend()
    plt.savefig()

    batch_size = params.model_training.model_params.batch_size
    _, _, test_dataset = datasets(batch_size)
    
    preds = model.predict(test_dataset)
    y_hat =np.argmax(preds, axis=1)
    y_true = np.concatenate([y for x, y in test_dataset], axis=0)
    y_true =np.argmax(y_true, axis=1)
    plot_confusion_matrix(y_true, y_hat)
    plt.savefig()