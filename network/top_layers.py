from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.layers import Lambda, Activation
from keras.models import Model
from network.rnn import TimeDistributed_CuDNNLSTM as TD_BiLSTM
from network.rnn import Bidirectional_CuDNNLSTM as BiLSTM
from keras_layer_normalization import LayerNormalization
import keras.backend as K


# 增加新层
def add_new_last_layer(base_model, nb_classes, train_mode=0):
    x = base_model.output
    if train_mode == 0:
        x = Flatten()(x)
        x = Dense(4096, activation='relu', kernel_initializer='he_normal')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu', kernel_initializer='he_normal')(x)
        x = Dropout(0.5)(x)
        x = Dense(nb_classes, activation='softmax', kernel_initializer='he_normal')(x)
    elif train_mode == 1:
        x = GlobalAveragePooling2D()(x)
        x = Dense(4096, activation='relu', kernel_initializer='he_normal')(x)
        x = Dropout(0.5)(x)
        x = Dense(nb_classes, activation='softmax', kernel_initializer='he_normal')(x)
    else:
        x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_1')(x)
        x = TD_BiLSTM(x, output_size=1024, name='lstm_1', mode='sum')
        x = LayerNormalization()(x)

        x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_2')(x)
        x = TD_BiLSTM(x, output_size=1024, name='lstm_2', mode='sum')
        x = LayerNormalization()(x)

        x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_3')(x)
        x = TD_BiLSTM(x, output_size=nb_classes, name='lstm_3', mode='sum')
        x = LayerNormalization()(x)

        x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_4')(x)
        x = TD_BiLSTM(x, output_size=nb_classes, name='lstm_4', mode='sum')
        x = LayerNormalization()(x)

        x = Activation('softmax', name='classification_out')(x)

    model = Model(inputs=base_model.input, outputs=x)

    return model
