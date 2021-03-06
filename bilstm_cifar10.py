from keras.utils import to_categorical
from keras.datasets import cifar10
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from network.bilstm_classification import BiLSTM_Sequential_Classification
from network.bilstm_classification import BiLSTM_Single_Classification
import os
import numpy as np
from helper import parser

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
    return x.astype('float32') / 255


def label_preprocess(label):
    y = np.empty((label.shape[0], input_size, input_size, 1))
    for i in range(label.shape[0]):
        y[i] = label[i]
        # print(y[i])
    return to_categorical(y, class_number)


def train(model, x_train, y_train, x_val, y_val):
    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 设置模型按什么标准进行保存。比如：acc,loss
    CP = ModelCheckpoint(ModelFile, monitor='val_acc',
                         verbose=1, save_best_only=True, mode='auto')
    # 设置如果性能不上升，停止学习
    ES = EarlyStopping(monitor='val_acc', patience=Patience)
    callbacks_list = [CP, ES]

    # 训练模型
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH,
              callbacks=callbacks_list, validation_data=(x_val, y_val))
    return model


def test(model, x_test, y_test):
    test_number = y_test.shape[0]
    y_pred = model.predict(x_test)

    y = np.empty((test_number, input_size, input_size, 1)).astype('int32')
    # y_final = np.full_like(y_test, 0)
    count = 0
    for num in range(test_number):
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

    # 数据预处理
    x_train = sample_preprocess(x_train)
    x_test = sample_preprocess(x_test)

    # input('stop')
    # 构建模型
    if Sequence:
        print('Start sequential training model...')
        y_train_sequence = label_preprocess(y_train)
        y_test_sequence = label_preprocess(y_test)
        model = BiLSTM_Sequential_Classification(
            input_shape=(input_size, input_size, input_channel),
            classes=class_number)
        model = train(model, x_train, y_train_sequence, x_test, y_test_sequence)
        # 测试模型
        acc = test(model, x_test, y_test)
    else:
        print('Start single training model...')
        y_train_single = to_categorical(y_train, class_number)
        y_test_single = to_categorical(y_test, class_number)
        model = BiLSTM_Single_Classification(
            input_shape=(input_size, input_size, input_channel),
            classes=class_number)
        model = train(model, x_train, y_train_single, x_test, y_test_single)
        acc = model.evaluate(x_test, y_test_single)[1]

    # score = model.evaluate(x_test, y_test)
    print('Accuracy is {}.'.format(acc))
    print('Congratulation! It finished.')


if __name__ == '__main__':

    args = parser.define_parser()

    Sequence = False
    if args.seq:
        Sequence = args.seq

    BATCH_SIZE = 32
    if args.bs:
        BATCH_SIZE = args.bs

    EPOCH = 1
    if args.ep:
        EPOCH = args.ep

    # 设置多少次不提升，就停止训练
    Patience = 10
    input_size = 32
    input_channel = 3
    class_number = 10
    dataset = 'cifar-10'
    ModelFile = model_path(dataset)
    main()