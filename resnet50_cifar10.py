from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D
from keras.layers import BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.models import Model
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.datasets import cifar10
import keras.backend as K

K.set_image_data_format('channels_last')
K.set_learning_phase(1)


def identity_block(X, f, filters, stage, block):
    # 定义基本名称
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    F1, F2, F3 = filters
    X_shortcut = X

    # 模块一：卷积、批标准化、激活函数
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1),
               padding="valid", name=conv_name_base + "2a",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    X = Activation("relu")(X)

    # 模块二：卷积、批标准化、激活函数
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1),
               padding="same", name=conv_name_base + "2b",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = Activation("relu")(X)

    # 模块三：卷积、批标准化
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1),
               padding="valid", name=conv_name_base + "2c",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)

    # 与捷径通道相加、激活函数
    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)
    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    # 定义基本名称
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters
    X_shortcut = X

    # 模块一：卷积、批标准化、激活函数
    X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a',
               padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # 模块二：卷积、批标准化、激活函数
    X = Conv2D(F2, (f, f), strides=(1, 1), name=conv_name_base + '2b',
               padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # 模块三：卷积、批标准化
    X = Conv2D(F3, (1, 1), strides=(1, 1), name=conv_name_base + '2c',
               padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # 捷径通道上的卷积，批标准化
    X_shortcut = Conv2D(F3, (1, 1), strides=(s, s),
                        name=conv_name_base + '1', padding='valid',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # 与捷径通道相加、激活函数
    X = layers.add([X, X_shortcut])
    X = Activation('relu')(X)
    return X


def ResNet50(input_shape=(32, 32, 3), classes=10):
    # 定义输入张量的形状
    X_input = Input(input_shape)

    # 零填充
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name="conv",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name="bn_conv1")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block="a", s=1)
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block="b")
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block="c")

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block="a", s=1)
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="b")
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="c")
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="d")

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block="a", s=2)
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="b")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="c")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="d")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="e")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="f")

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block="a", s=2)
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block="b")
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block="c")

    # 平均值池化
    X = AveragePooling2D(pool_size=(2, 2), padding="same")(X)

    # 扁平化
    X = Flatten()(X)

    # 全连接层、输出
    X = Dense(classes, activation="softmax", name="fc" + str(classes),
              kernel_initializer=glorot_uniform(seed=0))(X)

    # 创建模型
    model = Model(inputs=X_input, outputs=X, name="ResNet50")

    return model


def main():
    # 加载数据
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # 数据预处理
    x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # 构建模型
    model = ResNet50(input_shape=(32, 32, 3), classes=10)

    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 训练模型
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH)

    # 测试模型
    score = model.evaluate(x_test, y_test)
    print('Accuracy is {}.'.format(score[1]))


if __name__ == '__main__':
    BATCH_SIZE = 512
    EPOCH = 300
    main()
