import re
import os
import cv2
import numpy as np


def normalize_mid_points(X, ndims=3, skip_midpoints=False):
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

    if ndims == 3:
        if skip_midpoints:
            return x_n.reshape(N, 51)
        else:
            return np.concatenate([x_n, mp], axis=1).reshape(N, 54)
    elif ndims == 2:
        if skip_midpoints:
            return x_n[:, :, :2].reshape(N, 34)
        else:
            return np.concatenate([x_n[:, :, :2], mp[:, :, :2]], axis=1).reshape(N, 36)
    else:
        raise("Select number of dimensions")


def read_data(name, normalize_mp='2D', skip_midpoints=False):
    """ Read numpy array with saved keypoints

    Params:
        name - file name without .npy extension 
        normalize_mp - '3D' to keep scores, '2D' - keep (x,y) only
        skip_midpoints - add or omit midpoint coords

    Returns:
        tuple of normalized coords and array of labels
    """
    
    with open(f'data/labels/{name}') as f:
        labels = f.readlines()
    
    N = int(re.findall(r'\d+', labels[0])[0])
    X = np.load(f'data/keypoints/{name}.npy')
    X = X.reshape((N, 17, 3))
    y = np.zeros(N, dtype=int)
    
    for lab in labels:
        C = re.findall(r'\d:', lab)

        if len(C) == 1:
            C = int(C[0][0])
            idxs = re.findall(r'\d+-\d+', lab)
            for idx in idxs:
                start, stop = idx.split('-')
                y[int(start): int(stop)] = C
    if normalize_mp == '3D':
        X = normalize_mid_points(X=X, ndims=3, skip_midpoints=skip_midpoints)
    elif normalize_mp == '2D':
        X = normalize_mid_points(X=X, ndims=2, skip_midpoints=skip_midpoints)
    else:
        X = X.reshape(N, 51)
        
    print(X.shape)
    return X, y


def cart2pol(cart2pol):
    """Convert cartesian coords to polar
    
    Params:
        cart2pol - coords (x, y)
        
    Returns:
        polar coords (rho, phi)
    """
    rho = np.sqrt(cart2pol[0] ** 2 + cart2pol[1] ** 2)
    phi = np.arctan2(cart2pol[1], cart2pol[0])
    return(rho, phi)


def apply_cart2pol_along_axis(row):
    """Apply cart2pol to passed array of data
    
    Params:
        row - array of data, should have even length
        
    Returns:
        array of data of same length as input array of data
    """
    row_splitted_by_2 = row.reshape(-1, 2)
    return np.apply_along_axis(cart2pol, axis=1, arr=row_splitted_by_2).reshape(34,)


def format_feature_names(names, ndims=3, skip_midpoints=False):
    """"Form list of feature names

    Params:
        names - list of KEYPOINT_DICT keys
        ndims - number of dimensions (x, y, score)
        skip_midpoints - add or omit midpoint coords (x, y, score)

    Returns:
        list of feature names
    """
    feature_names = []
    for name in names:
        feature_names.append(name+'-x')
        feature_names.append(name+'-y')
        if ndims==3:
            feature_names.append(name+'-z')
    if not skip_midpoints:
        feature_names.append('mid_point-x')
        feature_names.append('mid_point-y')
        if ndims==3:
            feature_names.append('mid_point-z')
    return feature_names


def rotate(image, angle):
    """"Rotate image without cropping it

    Params:
        image - image array in BGR
        angle - angle to rotate image
    
    Returns:
        image array in BGR
    """
    h, w, _ = image.shape
    (cX, cY) = (w // 2, h // 2)

    rotMat = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)

    cos = np.abs(rotMat[0, 0])
    sin = np.abs(rotMat[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    rotMat[0, 2] += (nW / 2) - cX
    rotMat[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, rotMat, (nW, nH))


def save_tflite_model(tflite_model, dir='.'):
    """"Save tflite model to disk

    Params:
        tflite_model - tflite model
        dir - name of the directory containing models
    """
    model_names = [i for i in filter(lambda filename: '.tflite' in filename, os.listdir(dir))]
    model_num = 1
    model_nums = []
    
    for model_name in model_names:
        num = model_name[5:len(model_name) - 7]
        
        if num:
            model_nums.append(int(num))
            
    if len(model_nums):
        model_num = max(model_nums) + 1
    
    with open(f'{dir}/model{model_num}.tflite', 'wb') as f:
            f.write(tflite_model)
