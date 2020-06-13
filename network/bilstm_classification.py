from keras.layers import Input, Lambda, Activation, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Conv2D, Add, BatchNormalization, Concatenate, Dense
from keras.models import Model
import keras.backend as K
from .rnn import TimeDistributed_CuDNNLSTM as TD_BiLSTM
from .rnn import Bidirectional_CuDNNLSTM as BiLSTM
from keras_layer_normalization import LayerNormalization
from .rnn import Bidirectional_CuDNNLSTM as BiLSTM
from .rnn import TimeDistributed_ConvLSTM as TD_ConvLSTM
from .rnn import TimeDistributed_CuDNNGRU as TD_GRU
from .rnn import TimeDistributed_RNN as TD_RNN


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


def BiLSTM_Transpose_Layers(x, channels, name, mode='concat'):
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name=name + 'tran_1')(x)
    x = TD_GRU(x, output_size=channels, name=name + 'gru_1', mode=mode)
    x = LayerNormalization()(x)
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name=name + 'tran_2')(x)
    x = TD_GRU(x, output_size=channels, name=name + 'gru_2', mode=mode)
    x = LayerNormalization()(x)
    return x


def BiLSTM_Dense_Block(x, channels, name, num_sub_block=3):
    for i in range(num_sub_block):
        x_shortcut = x
        x = BiLSTM_Transpose_Layers(x, channels, name + str(i + 1) + '_')
        x = Concatenate()([x, x_shortcut])
    return x


def BiLSTM_Deep_V_0_6(input_shape, classes, num_block=2, repeat=1, sequence=True):
    inputs = Input(shape=input_shape)
    x = inputs

    for i in range(num_block):
        for j in range(repeat):
            x = BiLSTM_Dense_Block(x, channels=32 * (i + 1), name='Block_' + str(i * repeat + j + 1) + '_')
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='Pool_' + str(i + 1))(x)

    if sequence:
        x = BiLSTM_Transpose_Layers(x, channels=classes, name='output_', mode='sum')
        x = Activation('softmax', name='classification_out')(x)
    else:
        x = GlobalAveragePooling2D()(x)
        x = Dense(classes,
                  activation='softmax',
                  kernel_initializer='he_normal')(x)
    model = Model(inputs=inputs, outputs=x)
    return model


def RNN_Transpose_Layers(x, channels, name, mode='concat'):
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name=name + 'tran_1')(x)
    x = TD_RNN(x, output_size=channels, name=name + 'rnn_1', mode=mode)
    x = LayerNormalization()(x)
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name=name + 'tran_2')(x)
    x = TD_RNN(x, output_size=channels, name=name + 'rnn_2', mode=mode)
    x = LayerNormalization()(x)
    return x


def RNN_Dense_Block(x, channels, name, num_sub_block=3):
    for i in range(num_sub_block):
        x_shortcut = x
        x = RNN_Transpose_Layers(x, channels, name + str(i + 1) + '_')
        x = Concatenate()([x, x_shortcut])
    return x


def RNN_Deep_V_0_7(input_shape, classes, num_block=2, repeat=1, sequence=True):
    inputs = Input(shape=input_shape)
    x = inputs

    for i in range(num_block):
        for j in range(repeat):
            x = RNN_Dense_Block(x, channels=32 * (i + 1), name='Block_' + str(i * repeat + j + 1) + '_')
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='Pool_' + str(i + 1))(x)

    if sequence:
        x = RNN_Transpose_Layers(x, channels=classes, name='output_', mode='sum')
        x = Activation('softmax', name='classification_out')(x)
    else:
        x = GlobalAveragePooling2D()(x)
        x = Dense(classes,
                  activation='softmax',
                  kernel_initializer='he_normal')(x)
    model = Model(inputs=inputs, outputs=x)
    return model
