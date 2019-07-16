from keras.layers import Bidirectional, TimeDistributed, CuDNNLSTM
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
