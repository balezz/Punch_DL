import numpy as np
import os
import re


def normalize_mid_points(X, skip_midpoints=False):
    """Calculate middle point between two hips
        and substract this point from other coordinates

    Params:
        X - array of shape (Frames, KeyPoints, Dim),
            Frames - number of extracted video frames,
            KeyPoints = 17
            Dim = 3 (x, y, score)
        ndims - number of dimensions (x, y, score)
        skip_midpoints - add or omit midpoint coords (x, y, score)

    Returns:
        normalized coords
    """
    left_hip, right_hip = 11, 12
    N = X.shape[0]
    mid_points = (X[:, left_hip, :] + X[:, right_hip, :]) / 2
    mp = mid_points.reshape(N, 1, 3)
    x_n = X - mp

    if skip_midpoints:
        return x_n[:, :, :2].reshape(N, -1)
    else:
        return np.concatenate([x_n[:, :, :2], mp[:, :, :2]], axis=1).reshape(N, -1)


def reverse_labels(labels):
    """ Change right and left punch labels to correctly label mirrored videos
    
    Params:
        labels - numpy array of punch labels

    Returns:
        reversed labels
    """
    unique = [i for i in np.unique(labels) if i != 0]
    max_val = max(unique)
    labels[labels != 0] = labels[labels != 0] + 1
    labels[labels == max_val + 1] = labels[labels == max_val + 1] - 2
    return labels


# check that if video was reversed, labels were reversed as well
labels = np.array([0, 4, 5, 4, 4, 0, 0])
labels_reversed = reverse_labels(labels.copy())
assert np.sum(np.abs(labels - labels_reversed)) == np.sum(labels != 0)


def read_data(name, skip_midpoints=False, preprocess_data=None, all_labels=False):
    """ Read numpy array with saved keypoints

    Params:
        name - file name with or without .npy extension
        normalize_mp - '3D' to keep scores, '2D' - keep (x,y) only
        skip_midpoints - add or omit midpoint coords

    Returns:
        tuple of normalized coords and array of labels
    """
    name = name[:-4] if '.npy' in name else name

    is_reversed = 'reversed' in name

    if is_reversed:
        label_name = name.replace('_reversed', '')
    else:
        label_name = name
    
    with open(f'data/labels/{label_name}') as f:
        labels = f.readlines()
    
    X = np.load(f'data/keypoints/{name}.npy')
    N = X.shape[0]
    X = X.reshape((N, 17, 3))
    y = np.zeros(N, dtype=int)
    
    # first two lines label weak punches
    # next two lines label strong punches
    punch_line_num = 0
    
    for lab in labels:
        C = re.findall(r'\d:', lab)

        if len(C) == 1:
            C = int(C[0][0])  # label
            if all_labels and punch_line_num > 1:
                C = C + 6  # strong punches labels = weak punches label + 6

            idxs = re.findall(r'\d+-\d+', lab)
            for idx in idxs:
                start, stop = idx.split('-')
                y[int(start): int(stop)] = C
            
            punch_line_num = punch_line_num + 1

    if preprocess_data:
        X = preprocess_data(X)

    X = normalize_mid_points(X, skip_midpoints)
        
    print(label_name, f'reversed: {is_reversed}', f'data shape: ({X.shape[0]}, {X.shape[1]})', sep='|')
    print('-' * 20)

    if reversed:
        y = reverse_labels(y)
        
    return X, y


def get_train_data(skip_midpoints=False, preprocess_data=None):
    """ Get data for training and testing
    
    Returns:
        X_train, y_train, X_test, y_test
    """
    labels = os.listdir('data/labels')

    labels_by_punch_types = [[], [], []]

    # sort labels so that hooks go first, then jabs, then upers
    # significantly improves model accuracy
    for label in labels:
        if 'hook' in label:
            labels_by_punch_types[0].append(label)
        elif 'jab' in label:
            labels_by_punch_types[1].append(label)
        else:
            labels_by_punch_types[2].append(label)

    X_train_list = []
    y_train_list = []

    for labels in labels_by_punch_types:
        for label in labels:
            X, y = read_data(label, skip_midpoints, preprocess_data)
            X_train_list.append(X)
            y_train_list.append(y)

    X_train = np.concatenate(X_train_list)
    y_train = np.concatenate(y_train_list)
    
    return X_train, y_train