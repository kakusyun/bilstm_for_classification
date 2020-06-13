from keras.layers import Bidirectional, TimeDistributed, CuDNNLSTM
from keras.layers import ConvLSTM2D, CuDNNGRU, SimpleRNN
from keras.regularizers import l2


def TimeDistributed_CuDNNLSTM(inputs, output_size, name, mode, sequences=True):
    x = TimeDistributed(Bidirectional(CuDNNLSTM(output_size,
                                                return_sequences=sequences,
                                                kernel_initializer='he_normal',
                                                kernel_regularizer=l2(1e-4),
                                                name=name), merge_mode=mode))(inputs)
    return x


def Bidirectional_CuDNNLSTM(inputs, output_size, name, mode, sequences=True):
    x = Bidirectional(CuDNNLSTM(output_size,
                                return_sequences=sequences,
                                kernel_initializer='he_normal',
                                kernel_regularizer=l2(1e-4),
                                name=name), merge_mode=mode)(inputs)
    return x


def TimeDistributed_ConvLSTM(inputs, output_size, name, mode, sequences=True):
    x = TimeDistributed(Bidirectional(ConvLSTM2D(output_size, kernel_size=(3, 3), padding='same',
                                                 return_sequences=sequences,
                                                 kernel_initializer='he_normal',
                                                 kernel_regularizer=l2(1e-4),
                                                 name=name), merge_mode=mode))(inputs)
    return x


def TimeDistributed_CuDNNGRU(inputs, output_size, name, mode, sequences=True):
    x = TimeDistributed(Bidirectional(CuDNNGRU(output_size,
                                               return_sequences=sequences,
                                               kernel_initializer='he_normal',
                                               kernel_regularizer=l2(1e-4),
                                               name=name), merge_mode=mode))(inputs)
    return x


def TimeDistributed_RNN(inputs, output_size, name, mode, sequences=True):
    x = TimeDistributed(Bidirectional(SimpleRNN(output_size,
                                          return_sequences=sequences,
                                          kernel_initializer='he_normal',
                                          kernel_regularizer=l2(1e-4),
                                          unroll=True,
                                          name=name), merge_mode=mode))(inputs)
    return x
