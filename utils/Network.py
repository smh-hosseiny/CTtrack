import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dropout


def residual_block(y, nb_channels, _strides=(1, 1, 1), _kernel_size=(1, 1, 1), _project_shortcut=True):
    shortcut = y

    y = layers.Conv3D(nb_channels, kernel_size=_kernel_size, strides=_strides, padding='same')(y)

    if _project_shortcut or _strides != (1, 1, 1):
        shortcut = layers.Conv3D(nb_channels, kernel_size=(1, 1, 1), strides=_strides, padding='same')(shortcut)

    y = layers.add([shortcut, y])
    y = layers.ReLU()(y)
    return y
    

def network(grad_directions, output_size, dropout):

    inputs = keras.Input(shape=(3,3,3,grad_directions,),  name='diffusion_data')
    x = residual_block(inputs, 256)
    x = layers.Conv3D(512, kernel_size=(2,2,2))(x)
    x = residual_block(x, 512)
    x = layers.LayerNormalization()(x)
    x = layers.Flatten()(x)
    i = inputs[:,1,1,1]

    x2 = layers.Reshape((1,1,1,100))(i)
    x2 = residual_block(x2,704)
    x2 = layers.LayerNormalization()(x2)
    x2 = layers.Flatten()(x2)

    f = layers.Concatenate(axis=-1)([x,x2])
    f = layers.Reshape((40,40,3))(f)
    encoder = keras.Model(inputs = inputs, outputs = f, name="encoder")

    pi = keras.Input(shape=(40,40,3,),  name='encoded_data')
    x = layers.Conv2D(32, (3,3), activation = 'relu')(pi)
    x = layers.LayerNormalization()(x)
    x = layers.Conv2D(64, (3,3), activation = 'relu')(x)
    x = layers.Conv2D(64, (3,3), activation = 'relu')(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.MaxPool2D(pool_size=(2,2), strides = None)(x)
    x = layers.Conv2D(128, (3,3), activation = 'relu')(x)
    x = layers.Conv2D(128, (3,3), activation = 'relu')(x)
    x = layers.MaxPool2D(pool_size=(2,2), strides = None)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv2D(256, (3,3), activation = 'relu')(x)
    x = layers.Conv2D(256, (3,3), padding = 'same', activation = 'relu')(x)
    x = layers.Conv2D(512, (3,3), activation = 'relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(output_size, activation = 'linear')(x)
    predictor = keras.Model(inputs = pi, outputs = out, name="predictor")

    features = encoder(inputs)
    preds = predictor(features)
    model = keras.Model(inputs = inputs, outputs = preds, name="Model")

    return model
    
