import larq as lq
import tensorflow as tf

def create_bnn_model(input_shape=(30, 128, 128, 3)):
    """
    Constructs a Binary Neural Network (BNN) model for deepfake detection.

    Args:
        input_shape (tuple): Shape of the input video frames batch.

    Returns:
        tf.keras.Model: Compiled BNN model.
    """
    kwargs = dict(input_quantizer="ste_sign", kernel_quantizer="ste_sign", kernel_constraint="weight_clip")
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        lq.layers.QuantConv3D(32, (3, 3, 3), padding='same', **kwargs),
        tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1)),
        lq.layers.QuantConv3D(64, (3, 3, 3), padding='same', **kwargs),
        tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1)),
        tf.keras.layers.Flatten(),
        lq.layers.QuantDense(1, activation="sigmoid", **kwargs)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
