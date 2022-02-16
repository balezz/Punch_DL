import numpy as np
import tensorflow as tf
import tensorflow.keras as K

from utils import get_train_data


# load model
model = K.models.load_model('models/lstm__with_angles')

X_val, _ = get_train_data(skip_midpoints=True)
time_steps = 30


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()


# Run the model with TensorFlow to get expected results.
TEST_CASES = 10
x = tf.reshape(X_val, (-1, time_steps, 36))
# Run the model with TensorFlow Lite
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

for i in range(TEST_CASES):
    expected = model.predict(x[i:i+1])
    interpreter.set_tensor(input_details[0]["index"], x[i:i+1])
    interpreter.invoke()
    result = interpreter.get_tensor(output_details[0]["index"])

    # Assert if the result of TFLite model is consistent with the TF model.
    np.testing.assert_almost_equal(expected, result, decimal=5)
    print("Done. The result of TensorFlow matches the result of TensorFlow Lite.")

    # Please note: TfLite fused Lstm kernel is stateful, so we need to reset
    # the states.
    # Clean up internal states.
    interpreter.reset_all_variables()


# If everythins is ok, then save tflite model.
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)