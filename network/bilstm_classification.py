from keras.layers import Input, Lambda, Activation
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
