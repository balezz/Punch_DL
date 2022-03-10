import numpy as np
import tensorflow as tf
import tensorflow.keras as K

from utils import BASE_DIR


if __name__ == '__main__':
    # load model
    print(tf.__version__)
    model = K.models.load_model(BASE_DIR.joinpath('models', 'lstm__with_angles'))

    TEST_CASES = 10

    time_steps = 30
    keypoints = 36
    X_val = np.random.random((time_steps * TEST_CASES, keypoints))

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.inference_input_type = tf.float32
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()

    # Run the model with TensorFlow to get expected results.
    x = tf.reshape(X_val, (-1, time_steps, keypoints))

    # Run the model with TensorFlow Lite
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i in range(TEST_CASES):
        expected = model.predict(x[i:i + 1])
        interpreter.set_tensor(input_details[0]['index'], x[1:2])
        interpreter.invoke()
        result = interpreter.get_tensor(output_details[0]['index'])

        # Assert if the result of TFLite model is consistent with the TF model.
        np.testing.assert_almost_equal(expected, result, decimal=5)
        print('Done. The result of TensorFlow matches the result of TensorFlow Lite.')

        # Please note: TfLite fused Lstm kernel is stateful, so we need to reset
        # the states.
        # Clean up internal states.
        interpreter.reset_all_variables()

    # If everything is ok, then save tflite model.
    print('tests passed...')
    with open(BASE_DIR.joinpath('models', 'lstm__with_angles.tflite'), 'wb') as f:
        f.write(tflite_model)
