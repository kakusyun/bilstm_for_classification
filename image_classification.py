from keras.utils import to_categorical
from keras.datasets import cifar10
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
from helper import parser
from network import top_layers
from keras.preprocessing.image import ImageDataGenerator
from keras_applications.vgg16 import VGG16
from keras_applications.resnet_common import ResNet50
from keras_applications.imagenet_utils import preprocess_input
from keras.optimizers import Adagrad
import numpy as np
from tqdm import tqdm

K.set_image_data_format('channels_last')
K.set_learning_phase(1)


def out_model(dataset):
    ModelPath = './model/'
    if os.path.exists(ModelPath) is False:
        os.makedirs(ModelPath)
    # 保存的模型位置和名称，名称根据epoch和精度变化
    ModelFile = ModelPath + dataset + '-{epoch:03d}-{val_acc:.5f}-{val_loss:.5f}.hdf5'
    return ModelFile


def get_down_sampling(network):
    if network == 'vgg16':
        return 1 / 32
    else:
        return 1 / 32


def label_preprocess(label, down_sampling):
    output_size = int(input_size * down_sampling)
    y = np.empty((label.shape[0], output_size, output_size, 1))
    for i in range(label.shape[0]):
        y[i] = label[i]
        # print(y[i])
    return to_categorical(y, class_number)


# 生成模型
def get_model(train_model):
    if network == 'vgg16':
        weights_path = './model/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = VGG16(include_top=False, weights=weights_path,
                           input_shape=(input_size, input_size, input_channel))
    elif network == 'resnet50':
        weights_path = './model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = ResNet50(include_top=False, weights=weights_path,
                           input_shape=(input_size, input_size, input_channel))
    else:
        base_model = None
        print('Please select a base model.')
        os._exit(0)

    model = top_layers.add_new_last_layer(base_model, class_number, train_mode=train_model)
    model.summary()
    print('The base model has {} layers.'.format(len(base_model.layers)))
    print('The model has {} layers.'.format(len(model.layers)))
    return model, base_model


# 迁移学习的模型设置
def setup_to_transfer_learning(model, base_model):  # base_model
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])


def setup_to_all_layers_trainable(model, base_model):
    for layer in base_model.layers:
        layer.trainable = True
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])


# 微调模式的模型设置
def setup_to_fine_tune(model, base_model):
    GAP_LAYER = 10
    for layer in base_model.layers[:GAP_LAYER + 1]:
        layer.trainable = False
    for layer in base_model.layers[GAP_LAYER + 1:]:
        layer.trainable = True
    model.compile(optimizer=Adagrad(lr=0.0001), loss='categorical_crossentropy',
                  metrics=['accuracy'])


def train(model, base_model, x_train, y_train, x_val, y_val):
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                       rotation_range=30, width_shift_range=0.2,
                                       height_shift_range=0.2, shear_range=0.2,
                                       zoom_range=0.2, horizontal_flip=True)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_generator = train_datagen.flow(x_train, y_train, batch_size=32)
    val_generator = val_datagen.flow(x_val, y_val, batch_size=32)
    # datagen_train.fit(x_train)

    # 编译模型
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 设置模型按什么标准进行保存。比如：acc,loss
    CP = ModelCheckpoint(ModelFile, monitor='val_acc',
                         verbose=1, save_best_only=True, mode='auto')
    # 设置如果性能不上升，停止学习
    ES = EarlyStopping(monitor='val_acc', patience=Patience)
    callbacks_list = [CP, ES]

    # 训练模型
    setup_to_transfer_learning(model, base_model)
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=x_train.shape[0] // BATCH_SIZE,
                        epochs=2,
                        callbacks=callbacks_list,
                        validation_data=val_generator,
                        validation_steps=x_val.shape[0] // BATCH_SIZE)

    # setup_to_all_layers_trainable(model, base_model)
    setup_to_fine_tune(model, base_model)
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=x_train.shape[0] // BATCH_SIZE,
                        epochs=EPOCH,
                        callbacks=callbacks_list,
                        validation_data=val_generator,
                        validation_steps=x_val.shape[0] // BATCH_SIZE)
    return model


def test(model, x_test, y_test, down_sampling):
    y_pred = model.predict(x_test)

    test_number = y_test.shape[0]
    output_size = int(input_size * down_sampling)
    y = np.empty((test_number, output_size, output_size, 1)).astype('int32')
    # y_final = np.full_like(y_test, 0)
    count = 0
    for num in tqdm(range(test_number)):
        for i in range(y_pred.shape[1]):
            for j in range(y_pred.shape[2]):
                pred_cls = y_pred[num, i, j, :].tolist()
                y[num, i, j, 0] = pred_cls.index(max(pred_cls))

        if int(y_test[num]) == int(np.argmax(np.bincount(y[num, :, :, :].flatten()))):
            count += 1
    return count / test_number


def main():
    # 加载数据
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # 创建模型
    model, base_model = get_model(train_model)

    # input('stop')
    if train_model == 2:
        ds = get_down_sampling(network)
        y_train_seq = label_preprocess(y_train, ds)
        y_test_seq = label_preprocess(y_test, ds)

        model = train(model, base_model, x_train, y_train_seq, x_test, y_test_seq)

        x_test = preprocess_input(x_test)
        acc = test(model, x_test, y_test, ds)
    else:
        y_train = to_categorical(y_train, class_number)
        y_test = to_categorical(y_test, class_number)

        model = train(model, base_model, x_train, y_train, x_test, y_test)

        x_test = preprocess_input(x_test)
        acc = model.evaluate(x_test, y_test)[1]

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

    train_model = 2  # 0 标准模式 1 GAP模式 2 BiLSTM模式
    if args.tm:
        train_model = args.tm

    # 设置多少次不提升，就停止训练
    Patience = 50
    input_size = 32
    input_channel = 3
    class_number = 10
    dataset = 'cifar-10'
    network = 'vgg16'
    ModelFile = out_model(dataset)

    main()
