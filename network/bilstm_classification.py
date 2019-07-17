from keras.layers import Input, Lambda, Activation, MaxPooling2D, Conv2D, Add, BatchNormalization, Concatenate
from keras.models import Model
import keras.backend as K
from .rnn import TimeDistributed_CuDNNLSTM as TD_BiLSTM
from .rnn import Bidirectional_CuDNNLSTM as BiLSTM
from keras_layer_normalization import LayerNormalization


def BiLSTM_Sequential_Classification(input_shape, classes):
    inputs = Input(shape=input_shape)

    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_1')(inputs)
    x = TD_BiLSTM(x, output_size=512, name='lstm_1', mode='concat')
    x = LayerNormalization()(x)

    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_2')(x)
    x = TD_BiLSTM(x, output_size=512, name='lstm_2', mode='concat')
    x = LayerNormalization()(x)

    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_3')(x)
    x = TD_BiLSTM(x, output_size=classes, name='lstm_3', mode='sum')
    x = LayerNormalization()(x)

    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_4')(x)
    x = TD_BiLSTM(x, output_size=classes, name='lstm_4', mode='sum')
    x = LayerNormalization()(x)

    x = Activation('softmax', name='classification_out')(x)
    model = Model(inputs=inputs, outputs=x)
    model.summary()
    return model


def BiLSTM_Deep_Classification(input_shape, classes):
    inputs = Input(shape=input_shape)

    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_1')(inputs)
    x = TD_BiLSTM(x, output_size=64, name='lstm_1', mode='sum')
    x = Activation(activation='relu')(x)
    x = LayerNormalization()(x)
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_2')(x)
    x = TD_BiLSTM(x, output_size=64, name='lstm_2', mode='sum')
    x = Activation(activation='relu')(x)
    x = LayerNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_3')(x)
    x = TD_BiLSTM(x, output_size=128, name='lstm_3', mode='sum')
    x = Activation(activation='relu')(x)
    x = LayerNormalization()(x)
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_4')(x)
    x = TD_BiLSTM(x, output_size=128, name='lstm_4', mode='sum')
    x = Activation(activation='relu')(x)
    x = LayerNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_5')(x)
    x = TD_BiLSTM(x, output_size=256, name='lstm_5', mode='sum')
    x = Activation(activation='relu')(x)
    x = LayerNormalization()(x)
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_6')(x)
    x = TD_BiLSTM(x, output_size=256, name='lstm_6', mode='sum')
    x = Activation(activation='relu')(x)
    x = LayerNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_7')(x)
    x = TD_BiLSTM(x, output_size=classes, name='lstm_7', mode='sum')
    x = Activation(activation='relu')(x)
    x = LayerNormalization()(x)
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_8')(x)
    x = TD_BiLSTM(x, output_size=classes, name='lstm_8', mode='sum')
    x = Activation(activation='relu')(x)
    x = LayerNormalization()(x)

    x = Activation('softmax', name='classification_out')(x)
    model = Model(inputs=inputs, outputs=x)
    model.summary()
    return model


def BiLSTM_Deep_V_0_1(input_shape, classes):
    inputs = Input(shape=input_shape)

    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_1')(inputs)
    x = TD_BiLSTM(x, output_size=32, name='lstm_1', mode='sum')
    x = LayerNormalization()(x)
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_2')(x)
    x = TD_BiLSTM(x, output_size=32, name='lstm_2', mode='sum')
    x = LayerNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_3')(x)
    x = TD_BiLSTM(x, output_size=64, name='lstm_3', mode='sum')
    x = LayerNormalization()(x)
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_4')(x)
    x = TD_BiLSTM(x, output_size=64, name='lstm_4', mode='sum')
    x = LayerNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_5')(x)
    x = TD_BiLSTM(x, output_size=128, name='lstm_5', mode='sum')
    x = LayerNormalization()(x)
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_6')(x)
    x = TD_BiLSTM(x, output_size=128, name='lstm_6', mode='sum')
    x = LayerNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_7')(x)
    x = TD_BiLSTM(x, output_size=256, name='lstm_7', mode='sum')
    x = LayerNormalization()(x)
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_8')(x)
    x = TD_BiLSTM(x, output_size=256, name='lstm_8', mode='sum')
    x = LayerNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_9')(x)
    x = TD_BiLSTM(x, output_size=classes, name='lstm_9', mode='sum')
    x = LayerNormalization()(x)
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_10')(x)
    x = TD_BiLSTM(x, output_size=classes, name='lstm_10', mode='sum')
    x = LayerNormalization()(x)

    x = Activation('softmax', name='classification_out')(x)
    model = Model(inputs=inputs, outputs=x)
    model.summary()
    return model


def BiLSTM_Deep_V_0_2(input_shape, classes):
    inputs = Input(shape=input_shape)

    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_1')(inputs)
    x = TD_BiLSTM(x, output_size=64, name='lstm_1', mode='sum')
    x = LayerNormalization()(x)
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_2')(x)
    x = TD_BiLSTM(x, output_size=64, name='lstm_2', mode='sum')
    x = LayerNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_3')(x)
    x = TD_BiLSTM(x, output_size=128, name='lstm_3', mode='sum')
    x = LayerNormalization()(x)
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_4')(x)
    x = TD_BiLSTM(x, output_size=128, name='lstm_4', mode='sum')
    x = LayerNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_5')(x)
    x = TD_BiLSTM(x, output_size=256, name='lstm_5', mode='sum')
    x = LayerNormalization()(x)
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_6')(x)
    x = TD_BiLSTM(x, output_size=256, name='lstm_6', mode='sum')
    x = LayerNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_7')(x)
    x = TD_BiLSTM(x, output_size=512, name='lstm_7', mode='sum')
    x = LayerNormalization()(x)
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_8')(x)
    x = TD_BiLSTM(x, output_size=512, name='lstm_8', mode='sum')
    x = LayerNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_9')(x)
    x = TD_BiLSTM(x, output_size=classes, name='lstm_9', mode='sum')
    x = LayerNormalization()(x)
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_10')(x)
    x = TD_BiLSTM(x, output_size=classes, name='lstm_10', mode='sum')
    x = LayerNormalization()(x)

    x = Activation('softmax', name='classification_out')(x)
    model = Model(inputs=inputs, outputs=x)
    model.summary()
    return model


def BiLSTM_Deep_V_0_3(input_shape, classes):
    inputs = Input(shape=input_shape)

    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_1')(inputs)
    x = TD_BiLSTM(x, output_size=128, name='lstm_1', mode='sum')
    x = LayerNormalization()(x)
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_2')(x)
    x = TD_BiLSTM(x, output_size=128, name='lstm_2', mode='sum')
    x = LayerNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_3')(x)
    x = TD_BiLSTM(x, output_size=256, name='lstm_3', mode='sum')
    x = LayerNormalization()(x)
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_4')(x)
    x = TD_BiLSTM(x, output_size=256, name='lstm_4', mode='sum')
    x = LayerNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_7')(x)
    x = TD_BiLSTM(x, output_size=classes, name='lstm_7', mode='sum')
    x = LayerNormalization()(x)
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_8')(x)
    x = TD_BiLSTM(x, output_size=classes, name='lstm_8', mode='sum')
    x = LayerNormalization()(x)

    x = Activation('softmax', name='classification_out')(x)
    model = Model(inputs=inputs, outputs=x)
    model.summary()
    return model


def BiLSTM_Deep_V_0_4(input_shape, classes):
    inputs = Input(shape=input_shape)

    x_shortcut = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu',
                        kernel_initializer='he_normal')(inputs)

    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_1')(inputs)
    x = TD_BiLSTM(x, output_size=128, name='lstm_1', mode='sum')
    x = LayerNormalization()(x)
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_2')(x)
    x = TD_BiLSTM(x, output_size=128, name='lstm_2', mode='sum')
    x = LayerNormalization()(x)
    x = Add()([x, x_shortcut])
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x_shortcut = Conv2D(256, kernel_size=(1, 1), padding='same', activation='relu',
                        kernel_initializer='he_normal')(x)

    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_3')(x)
    x = TD_BiLSTM(x, output_size=256, name='lstm_3', mode='sum')
    x = LayerNormalization()(x)
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_4')(x)
    x = TD_BiLSTM(x, output_size=256, name='lstm_4', mode='sum')
    x = LayerNormalization()(x)
    x = Add()([x, x_shortcut])
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_7')(x)
    x = TD_BiLSTM(x, output_size=classes, name='lstm_7', mode='sum')
    x = LayerNormalization()(x)
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_8')(x)
    x = TD_BiLSTM(x, output_size=classes, name='lstm_8', mode='sum')
    x = LayerNormalization()(x)

    x = Activation('softmax', name='classification_out')(x)
    model = Model(inputs=inputs, outputs=x)
    model.summary()
    return model


def BiLSTM_Deep_V_0_5(input_shape, classes):
    inputs = Input(shape=input_shape)

    x_shortcut_1_1 = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu',
                            kernel_initializer='he_normal')(inputs)
    x_shortcut_1_1 = BatchNormalization()(x_shortcut_1_1)
    x_shortcut_1_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x_shortcut_1_1)

    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_1')(inputs)
    x = TD_BiLSTM(x, output_size=128, name='lstm_1', mode='sum')
    x = LayerNormalization()(x)
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_2')(x)
    x = TD_BiLSTM(x, output_size=128, name='lstm_2', mode='sum')
    x = LayerNormalization()(x)
    x = Concatenate()([x, x_shortcut_1_1])
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x_shortcut_2_1 = x

    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_3')(x)
    x = TD_BiLSTM(x, output_size=256, name='lstm_3', mode='sum')
    x = LayerNormalization()(x)
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_4')(x)
    x = TD_BiLSTM(x, output_size=256, name='lstm_4', mode='sum')
    x = LayerNormalization()(x)
    x = Concatenate()([x, x_shortcut_2_1, x_shortcut_1_2])
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_7')(x)
    x = TD_BiLSTM(x, output_size=classes, name='lstm_7', mode='sum')
    x = LayerNormalization()(x)
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_8')(x)
    x = TD_BiLSTM(x, output_size=classes, name='lstm_8', mode='sum')
    x = LayerNormalization()(x)

    x = Activation('softmax', name='classification_out')(x)
    model = Model(inputs=inputs, outputs=x)
    model.summary()
    return model


# def BiLSTM_Identity_Block():


def BiLSTM_Single_Classification(input_shape, classes):
    inputs = Input(shape=input_shape)

    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_1')(inputs)
    x = TD_BiLSTM(x, output_size=512, name='lstm_1', mode='concat')
    # x = LayerNormalization()(x)

    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_2')(x)
    x = TD_BiLSTM(x, output_size=512, name='lstm_2', mode='concat')
    # x = LayerNormalization()(x)

    # x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_3')(x)
    # x = TD_BiLSTM(x, output_size=classes, name='lstm_3', mode='sum')
    # x = LayerNormalization()(x)

    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_4')(x)
    x = TD_BiLSTM(x, output_size=classes, name='lstm_4', mode='sum', sequences=False)
    # x = LayerNormalization()(x)

    x = BiLSTM(x, output_size=classes, name='lstm_5', mode='sum', sequences=False)
    x = LayerNormalization()(x)

    x = Activation('softmax', name='classification_out')(x)
    model = Model(inputs=inputs, outputs=x)
    model.summary()
    return model
