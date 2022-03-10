import click
import cv2 as cv
import tensorflow.keras as K
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import seaborn as sns
import tensorflow as tf

from pathlib import PurePath
from tqdm import tqdm

from utils import normalize_mid_points, BASE_DIR


LABELS = [
    'no punch',

    'left jab weak',
    'right jab weak',
    'left hook weak',
    'right hook weak',
    'left uper weak',
    'right uper weak',
    
    'left jab strong',
    'right jab strong',
    'left hook strong',
    'right hook strong',
    'left uper strong',
    'right uper strong'
]

KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

COLOR_MAP = {
    'm': (255, 0, 255),
    'c': (0, 255, 255),
    'y': (255, 255, 0)
}


INPUT_SIZE = 192

# Confidence score to determine whether a keypoint prediction is reliable.
MIN_CROP_KEYPOINT_SCORE = 0.2


def download_file(url, filename):
    """Downloads file from url"""
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()


def get_movenet_interpreter():
    """Downloads, initializes and returns movenet model interpreter"""
    # check if tflite model is available
    if not os.path.exists(BASE_DIR.joinpath('models', 'movenet_model.tflite')):
        url = 'https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite'
        print('movenet model not found')
        print(f'start downloading model from {url}')
        download_file(url, BASE_DIR.joinpath('models', 'movenet_model.tflite'))

    # Initialize movenet model
    interpreter = tf.lite.Interpreter(model_path=BASE_DIR.joinpath('models', 'movenet_model.tflite').__str__())
    interpreter.allocate_tensors()
    return interpreter


def remove_movenet_model():
    """Deletes movenet model"""
    if os.path.exists(BASE_DIR.joinpath('models', 'movenet_model.tflite')):
        os.remove(BASE_DIR.joinpath('models', 'movenet_model.tflite'))


def get_punch_classifier_model(model_path):
    """Initializes and returns punch classifier model interpreter"""
    # Initialize punch classifier model
    return K.models.load_model(model_path)


def get_interpreter_results(interpreter, inputs):
    """Runs interpreter"""
    # TF Lite format expects tensor type of uint8.
    inputs = tf.cast(inputs, dtype=tf.uint8)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], inputs.numpy())
    # Invoke inference.
    interpreter.invoke()
    # Get the model prediction.
    return interpreter.get_tensor(output_details[0]["index"])


def init_crop_region(image_height, image_width):
    """Defines the default crop region.

    The function provides the initial crop region (pads the full image from both
    sides to make it a square image) when the algorithm cannot reliably determine
    the crop region from the previous frame.
    """
    if image_width > image_height:
        box_height = image_width / image_height
        box_width = 1.0
        y_min = (image_height / 2 - image_width / 2) / image_height
        x_min = 0.0
    else:
        box_height = 1.0
        box_width = image_height / image_width
        y_min = 0.0
        x_min = (image_width / 2 - image_height / 2) / image_width

    return {
        "y_min": y_min,
        "x_min": x_min,
        "y_max": y_min + box_height,
        "x_max": x_min + box_width,
        "height": box_height,
        "width": box_width,
    }


def torso_visible(keypoints):
    """Checks whether there are enough torso keypoints.

    This function checks whether the model is confident at predicting one of the
    shoulders/hips which is required to determine a good crop region.
    """
    return (
               keypoints[0, 0, KEYPOINT_DICT["left_hip"], 2] > MIN_CROP_KEYPOINT_SCORE
               or keypoints[0, 0, KEYPOINT_DICT["right_hip"], 2] > MIN_CROP_KEYPOINT_SCORE
           ) and (
               keypoints[0, 0, KEYPOINT_DICT["left_shoulder"], 2] > MIN_CROP_KEYPOINT_SCORE
               or keypoints[0, 0, KEYPOINT_DICT["right_shoulder"], 2] > MIN_CROP_KEYPOINT_SCORE
           )


def determine_torso_and_body_range(keypoints, target_keypoints, center_y, center_x):
    """Calculates the maximum distance from each keypoints to the center location.

    The function returns the maximum distances from the two sets of keypoints:
    full 17 keypoints and 4 torso keypoints. The returned information will be
    used to determine the crop size. See determineCropRegion for more detail.
    """
    torso_joints = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    max_torso_yrange = 0.0
    max_torso_xrange = 0.0
    for joint in torso_joints:
        dist_y = abs(center_y - target_keypoints[joint][0])
        dist_x = abs(center_x - target_keypoints[joint][1])
        if dist_y > max_torso_yrange:
            max_torso_yrange = dist_y
        if dist_x > max_torso_xrange:
            max_torso_xrange = dist_x

    max_body_yrange = 0.0
    max_body_xrange = 0.0
    for joint in KEYPOINT_DICT.keys():
        if keypoints[0, 0, KEYPOINT_DICT[joint], 2] < MIN_CROP_KEYPOINT_SCORE:
            continue
        dist_y = abs(center_y - target_keypoints[joint][0])
        dist_x = abs(center_x - target_keypoints[joint][1])
        if dist_y > max_body_yrange:
            max_body_yrange = dist_y

        if dist_x > max_body_xrange:
            max_body_xrange = dist_x

    return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]


def determine_crop_region(keypoints, image_height, image_width):
    """Determines the region to crop the image for the model to run inference on.

    The algorithm uses the detected joints from the previous frame to estimate
    the square region that encloses the full body of the target person and
    centers at the midpoint of two hip joints. The crop size is determined by
    the distances between each joints and the center point.
    When the model is not confident with the four torso joint predictions, the
    function returns a default crop which is the full image padded to square.
    """
    target_keypoints = {}
    for joint in KEYPOINT_DICT.keys():
        target_keypoints[joint] = [
            keypoints[0, 0, KEYPOINT_DICT[joint], 0] * image_height,
            keypoints[0, 0, KEYPOINT_DICT[joint], 1] * image_width,
        ]

    if torso_visible(keypoints):
        center_y = (target_keypoints["left_hip"][0] + target_keypoints["right_hip"][0]) / 2
        center_x = (target_keypoints["left_hip"][1] + target_keypoints["right_hip"][1]) / 2

        (
            max_torso_yrange,
            max_torso_xrange,
            max_body_yrange,
            max_body_xrange,
        ) = determine_torso_and_body_range(
            keypoints, target_keypoints, center_y, center_x
        )

        crop_length_half = np.amax(
            [
                max_torso_xrange * 1.9,
                max_torso_yrange * 1.9,
                max_body_yrange * 1.2,
                max_body_xrange * 1.2,
            ]
        )

        tmp = np.array([center_x, image_width - center_x, center_y, image_height - center_y])
        crop_length_half = np.amin([crop_length_half, np.amax(tmp)])

        crop_corner = [center_y - crop_length_half, center_x - crop_length_half]

        if crop_length_half > max(image_width, image_height) / 2:
            return init_crop_region(image_height, image_width)
        else:
            crop_length = crop_length_half * 2
            return {
                "y_min": crop_corner[0] / image_height,
                "x_min": crop_corner[1] / image_width,
                "y_max": (crop_corner[0] + crop_length) / image_height,
                "x_max": (crop_corner[1] + crop_length) / image_width,
                "height": (crop_corner[0] + crop_length) / image_height - crop_corner[0] / image_height,
                "width": (crop_corner[1] + crop_length) / image_width - crop_corner[1] / image_width,
            }
    else:
        return init_crop_region(image_height, image_width)


def crop_and_resize(image, crop_region, crop_size):
    """Crops and resize the image to prepare for the model input."""
    boxes = [
        [
            crop_region["y_min"],
            crop_region["x_min"],
            crop_region["y_max"],
            crop_region["x_max"],
        ]
    ]
    output_image = tf.image.crop_and_resize(
        image, box_indices=[0], boxes=boxes, crop_size=crop_size
    )
    return output_image


def run_inference(interpreter, movenet, image, crop_region, crop_size):
    """Runs model inferece on the cropped region.

    The function runs the model inference on the cropped region and updates the
    model output to the original image coordinate system.
    """
    image_height, image_width, _ = image.shape
    input_image = crop_and_resize(
        tf.expand_dims(image, axis=0), crop_region, crop_size=crop_size
    )
    # Run model inference.
    keypoints_with_scores = movenet(interpreter, input_image)
    # Update the coordinates.
    for idx in range(17):
        keypoints_with_scores[0, 0, idx, 0] = (
                  crop_region['y_min'] * image_height
                  + crop_region['height'] * image_height * keypoints_with_scores[0, 0, idx, 0]
          ) / image_height
        keypoints_with_scores[0, 0, idx, 1] = (
                  crop_region['x_min'] * image_width
                  + crop_region['width'] * image_width * keypoints_with_scores[0, 0, idx, 1]
          ) / image_width
    return keypoints_with_scores


def _keypoints_and_edges_for_display(keypoints_with_scores,
                                     height,
                                     width,
                                     keypoint_threshold=0.11):
    """Returns high confidence keypoints and edges for visualization.
    Args:
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    height: height of the image in pixels.
    width: width of the image in pixels.
    keypoint_threshold: minimum confidence score for a keypoint to be
      visualized.
    Returns:
    A (keypoints_xy, edges_xy, edge_colors) containing:
      * the coordinates of all keypoints of all detected entities;
      * the coordinates of all skeleton edges of all detected entities;
      * the colors in which the edges should be plotted.
    """
    keypoints_all = []
    keypoint_edges_all = []
    edge_colors = []

    num_instances, _, _, _ = keypoints_with_scores.shape

    for idx in range(num_instances):
        kpts_x = keypoints_with_scores[0, idx, :, 1]
        kpts_y = keypoints_with_scores[0, idx, :, 0]
        kpts_scores = keypoints_with_scores[0, idx, :, 2]
        kpts_absolute_xy = np.stack([width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
        kpts_above_thresh_absolute = kpts_absolute_xy[kpts_scores > keypoint_threshold, :]
        keypoints_all.append(kpts_above_thresh_absolute)

    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        if kpts_scores[edge_pair[0]] > keypoint_threshold and kpts_scores[edge_pair[1]] > keypoint_threshold:
            x_start = kpts_absolute_xy[edge_pair[0], 0]
            y_start = kpts_absolute_xy[edge_pair[0], 1]
            x_end = kpts_absolute_xy[edge_pair[1], 0]
            y_end = kpts_absolute_xy[edge_pair[1], 1]
            line_seg = np.array([[x_start, y_start], [x_end, y_end]])
            keypoint_edges_all.append(line_seg)
            edge_colors.append(color)

    if keypoints_all:
        keypoints_xy = np.concatenate(keypoints_all, axis=0)
    else:
        keypoints_xy = np.zeros((0, 17, 2))

    if keypoint_edges_all:
        edges_xy = np.stack(keypoint_edges_all, axis=0)
    else:
        edges_xy = np.zeros((0, 2, 2))
    return keypoints_xy, edges_xy, edge_colors


def draw_keypoints(frame, keypoints):
    height, width, _ = frame.shape

    points, edges, edge_colors = _keypoints_and_edges_for_display(keypoints, height, width)

    # draw points
    for p in points:
        px, py = p
        cv.circle(frame, (int(px), int(py)), 4, (0, 255, 0), -1)

    # draw edges
    for e, c in zip(edges, edge_colors):
        ex1, ey1 = e[0]
        ex2, ey2 = e[1]

        cv.line(frame, (int(ex1), int(ey1)), (int(ex2), int(ey2)), COLOR_MAP[c], 1)


def get_mode(x):
    vals, counts = np.unique(x, return_counts=True)
    index = np.argmax(counts)
    return vals[index]


def generate_histogram(predictions):
    """Generates histogram from predictions to visualize results of running the model"""
    # enable seaborn theming
    sns.set()
    plt.hist(predictions, bins=[i - 0.5 for i in range(len(LABELS) + 1)])
    plt.title('Histogram of predicted values')
    plt.xticks([i for i in range(len(LABELS))], labels=LABELS, rotation=45)
    plt.tight_layout()
    plt.show()


@click.command()
@click.option('--device', default=0, help='Device to capture video from (in case you have more than one)')
@click.option('--debug', is_flag=True, help='Enable debug')
@click.option('--model', default=PurePath('models', 'lstm_test_model').__str__(), help='Model to test')
def main(device, debug, model):
    if debug:
        print('debug is on')

    movenet_interpreter = get_movenet_interpreter()
    punch_classifier_model = get_punch_classifier_model(model)

    cap = cv.VideoCapture(device)
    buffer = []
    predictions = []

    ret, frame = cap.read()

    height, width, _ = frame.shape
    crop_region = init_crop_region(height, width)

    while cap.isOpened():
        keypoints_with_scores = run_inference(
            movenet_interpreter,
            get_interpreter_results,
            frame,
            crop_region,
            crop_size=[INPUT_SIZE, INPUT_SIZE]
        )

        buffer.append(keypoints_with_scores.reshape(17, 3))

        crop_region = determine_crop_region(
            keypoints_with_scores, height, width
        )

        if debug:
            # visualize keypoints and give more info on model predictions
            draw_keypoints(frame, keypoints_with_scores)

        if len(buffer) >= 120:
            buffer = buffer[-120:]
            punch_classifier_inputs = normalize_mid_points(np.array(buffer), skip_midpoints=True).reshape(4, 30, 34)
            label_scores = punch_classifier_model.predict(punch_classifier_inputs)[-1][-10:]  # get last 10 predictions
            prediction = get_mode(np.argmax(label_scores, axis=1))  # take mode of the last 10 predictions
            predictions.append(prediction)
            cv.putText(frame,
                       LABELS[prediction],
                       (width // 2 - 20, height - 50), cv.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 2)

        cv.imshow('test', frame)

        ret, frame = cap.read()

        if not ret:
            break

        if cv.waitKey(20) & 0xFF == ord('q'):
            break

    # remove_movenet_model()
    generate_histogram(predictions)


if __name__ == '__main__':
    predictions = main()

