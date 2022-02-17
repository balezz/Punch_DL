import click
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import seaborn as sns
import tensorflow as tf

from pathlib import PurePath
from tqdm import tqdm

from utils import normalize_mid_points


LABELS = [
    'no punch',

    'left jab weak',
    'right jab weak',
    'left hook weak',
    'right hook weak',
    'left uper weak',
    'right uper weak'
    
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
    if not os.path.exists('movenet_model.tflite'):
        url = 'https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite'
        print('movenet model not found')
        print(f'start downloading model from {url}')
        download_file(url, 'movenet_model.tflite')

    # Initialize movenet model
    interpreter = tf.lite.Interpreter(model_path="movenet_model.tflite")
    interpreter.allocate_tensors()
    return interpreter


def remove_movenet_model():
    """Deletes movenet model"""
    if os.path.exists('movenet_model.tflite'):
        os.remove('movenet_model.tflite')


def get_punch_classifier_interpreter(model_path=PurePath('models', 'models/model.tflite').__str__()):
    """Initializes and returns punch classifier model interpreter"""
    # Initialize punch classifier model
    punch_classifier_interpreter = tf.lite.Interpreter(model_path=model_path)
    punch_classifier_interpreter.allocate_tensors()
    return punch_classifier_interpreter


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
@click.option('--debug', default=False, help='Enable debug')
@click.option('--model', default=PurePath('models', 'model.tflite').__str__(), help='Model to test')
def main(device, debug, model):
    if debug:
        print('debug is on')

    movenet_interpreter = get_movenet_interpreter()
    punch_classifier_interpreter = get_punch_classifier_interpreter(model)

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
            pass

        if len(buffer) >= 30:
            label_scores = get_interpreter_results(
                        punch_classifier_interpreter,
                        normalize_mid_points(np.array(buffer[-30:])))[0][-1]
            prediction = np.argmax(label_scores)
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

    generate_histogram(predictions)
    remove_movenet_model()


if __name__ == '__main__':
    main()
