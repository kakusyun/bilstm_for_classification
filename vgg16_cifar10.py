from keras.utils import to_categorical
from keras.datasets import cifar10
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
from helper import parser
from network import vggnet
from keras.preprocessing.image import ImageDataGenerator

K.set_image_data_format('channels_last')
K.set_learning_phase(1)


def model_path(dataset):
    ModelPath = './model/'
    if os.path.exists(ModelPath) is False:
        os.makedirs(ModelPath)
    # 保存的模型位置和名称，名称根据epoch和精度变化
    ModelFile = ModelPath + dataset + '-{epoch:03d}-{val_acc:.5f}-{val_loss:.5f}.hdf5'
    return ModelFile


def sample_preprocess(x):
    # x = x.reshape(x.shape[0], input_size, input_size, input_channel)
    x = x.astype('float32') / 255
    return x


def train(model, x_train, y_train, x_val, y_val):
    datagen_train = ImageDataGenerator(rotation_range=30, width_shift_range=0.2,
                                       height_shift_range=0.2, shear_range=0.2,
                                       zoom_range=0.2, horizontal_flip=True)
    train_generator = datagen_train.flow(x_train, y_train, batch_size=32)

    # datagen_train.fit(x_train)

    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 设置模型按什么标准进行保存。比如：acc,loss
    CP = ModelCheckpoint(ModelFile, monitor='val_acc',
                         verbose=1, save_best_only=False, mode='auto')
    # 设置如果性能不上升，停止学习
    ES = EarlyStopping(monitor='val_acc', patience=Patience)
    callbacks_list = [CP, ES]

    # 训练模型
    model.fit_generator(generator=train_generator,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCH,
                        callbacks=callbacks_list,
                        validation_data=(x_val, y_val))
    return model


def main():
    # 加载数据
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # class_number = len(np.unique(y_train))
    # 数据预处理
    x_train = sample_preprocess(x_train)
    x_test = sample_preprocess(x_test)

    # input('stop')
    # 构建模型
    y_train = to_categorical(y_train, class_number)
    y_test = to_categorical(y_test, class_number)

    model = vggnet.VGG16(input_shape=(input_size, input_size, input_channel),
                         classes=class_number)

    model = train(model, x_train, y_train, x_test, y_test)
    acc = model.evaluate(x_test, y_test)[1]

    # score = model.evaluate(x_test, y_test)
    print('Accuracy is {}.'.format(acc))
    print('Congratulation! It finished.')


if __name__ == '__main__':

    args = parser.define_parser()

    BATCH_SIZE = 512
    if args.bs:
        BATCH_SIZE = args.bs

    EPOCH = 300
    if args.ep:
        EPOCH = args.ep

    # 设置多少次不提升，就停止训练
    Patience = 50
    input_size = 32
    input_channel = 3
    class_number = 10
    dataset = 'cifar-10'
    ModelFile = model_path(dataset)
    main()
