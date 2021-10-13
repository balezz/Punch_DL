import numpy as np
import matplotlib.pyplot as plt

def random_split_train_val(X, y, num_val, seed=42):
    """
    X - features
    y - labels
    num_val - number of validation sample
    seed=42 - random seed
    """
    np.random.seed(seed)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    train_indices = indices[:-num_val]
    train_X = X[train_indices]
    train_y = y[train_indices]

    val_indices = indices[-num_val:]
    val_X = X[val_indices]
    val_y = y[val_indices]

    return train_X, train_y, val_X, val_y


def visualize_confusion_matrix(confusion_matrix):
    assert confusion_matrix.shape[0] == confusion_matrix.shape[1]
    size = confusion_matrix.shape[0]
    fig = plt.figure(figsize=(10,10))
    plt.title("Confusion matrix")
    plt.xlabel("predicted")
    plt.ylabel("ground truth")
    res = plt.imshow(confusion_matrix, cmap='GnBu', interpolation='nearest')
    cb = fig.colorbar(res)
    plt.xticks(np.arange(size))
    plt.yticks(np.arange(size))
    for i, row in enumerate(confusion_matrix):
        for j, count in enumerate(row):
            plt.text(j, i, count, fontsize=14, horizontalalignment='center', verticalalignment='center')
    
    
def build_confusion_matrix(predictions, ground_truth, n_classes=5):
    """
    Builds confusion matrix from predictions and ground truth

    predictions: np array of ints, model predictions for all validation samples
    ground_truth: np array of ints, ground truth for all validation samples
    
    Returns:
    np array of ints, (n, n), counts of samples for predicted/ground_truth classes
    """
    
    confusion_matrix = np.zeros((n_classes, n_classes), np.int)
    for i in range(len(predictions)):
        confusion_matrix[ground_truth[i]][predictions[i]] += 1
    
    return confusion_matrix

def calc_metrics(confusion_matrix):
    n = len(confusion_matrix)
    precision = np.zeros(n)
    recall = np.zeros(n)
    f_1 = np.zeros(n)
    
    for i in range(n):
        TP = confusion_matrix[i, i]
        precision[i] = TP / np.sum(confusion_matrix[:, i])
        recall[i] = TP / np.sum(confusion_matrix[i, :])
        f_1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
    
    return {'precision': precision, 'recall': recall, 'F1-score':f_1}


