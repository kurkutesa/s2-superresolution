from keras.models import Model, Input
from keras.layers import Conv2D, Concatenate, Activation, Lambda, Add
import keras.backend as K

K.set_image_data_format('channels_first')


def resBlock(x, channels, kernel_size=[3, 3], scale=0.1):
    """
    This method is a series of layers that operate on input image x and returns an output
    image tmp and in the final step adds this output image to the initial input image x.

    :param x: Input image
    :param channels: Number of output filters in the convolution
    :param kernel_size: The height and width of the 2D convolution window
    :param scale: float number define the the value for residual scaling
    """
    tmp = Conv2D(channels, kernel_size, kernel_initializer='he_uniform', padding='same')(x)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(channels, kernel_size, kernel_initializer='he_uniform', padding='same')(tmp)
    tmp = Lambda(lambda x: x * scale)(tmp)

    return Add()([x, tmp])


def s2model(input_shape, num_layers=32, feature_size=256):
    """
    This method essentially is based on the Keras library to create and return a custom convolutional neural network.

    :param input_shape: The shape of the input image
    :param num_layers: The number of resBlocks
    :param feature_size: The number of filters

    """

    input10 = Input(shape=input_shape[0])
    input20 = Input(shape=input_shape[1])
    if len(input_shape) == 3:
        input60 = Input(shape=input_shape[2])
        x = Concatenate(axis=1)([input10, input20, input60])
    else:
        x = Concatenate(axis=1)([input10, input20])

    # Treat the concatenation
    x = Conv2D(feature_size, (3, 3), kernel_initializer='he_uniform', activation='relu', padding='same')(x)

    for i in range(num_layers):
        x = resBlock(x, feature_size)

    # One more convolution, and then we add the output of our first conv layer
    x = Conv2D(input_shape[-1][0], (3, 3), kernel_initializer='he_uniform', padding='same')(x)
    # x = Dropout(0.3)(x)
    if len(input_shape) == 3:
        x = Add()([x, input60])
        model = Model(inputs=[input10, input20, input60], outputs=x)
    else:
        x = Add()([x, input20])
        model = Model(inputs=[input10, input20], outputs=x)
    return model

