
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import mean_squared_error as MSE
from load_params import load_params
from image_pipeline import datasets

%matplotlib inline

def plot_confusion_matrix(y_true, y_pred):
    confusion_matrix_obj = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(confusion_matrix_obj, annot=True, fmt='d', linewidths=.75,  cbar=False, ax=ax,cmap='Blues',linecolor='white')
    #  square=True,
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')


def plot_model_metrics(model, hist):
    params = load_params('params.yaml')
    batch_size = params.model_training.model_params.batch_size
    _, _, test_dataset = datasets(batch_size)
    
    test_loss, test_acc, test_f1 = model.evaluate(test_dataset)

    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_acc}')
    print(f'Test F1: {test_f1}')

    losses = hist.history['loss']
    accs = hist.history['accuracy']
    val_losses = hist.history['val_loss']
    val_accs = hist.history['val_accuracy']
    
    # length of any hist value = number of epochs
    epochs = len(losses)

    plt.figure(figsize=(16, 4))
    for i, metrics in enumerate(zip([losses, accs], [val_losses, val_accs], ['Loss', 'Accuracy'])):
        plt.subplot(1, 2, i + 1)
        plt.plot(range(epochs), metrics[0], label=f'Training {metrics[2]}')
        plt.plot(range(epochs), metrics[1], label=f'Validation {metrics[2]}')
        plt.legend()
    plt.show()
    preds = model.predict(test_dataset)
    y_hat =np.argmax(preds, axis=1)
    y_true = np.concatenate([y for x, y in test_dataset], axis=0)
    y_true =np.argmax(y_true, axis=1)
    X_test = np.concatenate([x for x, y in test_dataset], axis=0)
    plot_confusion_matrix(y_true, y_hat)
    plt.show()