from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os
from network.bilstm_classification import BiLSTM_Sequential_Classification
from network.bilstm_classification import BiLSTM_Deep_Classification
from network.bilstm_classification import BiLSTM_Deep_V_0_1
from network.bilstm_classification import BiLSTM_Deep_V_0_2
from network.bilstm_classification import BiLSTM_Deep_V_0_3
from network.bilstm_classification import BiLSTM_Deep_V_0_4
from network.bilstm_classification import BiLSTM_Deep_V_0_5
from network.bilstm_classification import BiLSTM_Deep_V_0_6

from helper.parser import define_parser
from keras.applications.densenet import DenseNet121
from tqdm import tqdm
from helper import handle_model as hm


def label_preprocess(label, down_sampling):
    output_size = int(input_size * down_sampling)
    y = np.empty((label.shape[0], output_size, output_size, 1))
    for i in range(label.shape[0]):
        y[i] = label[i]
        # print(y[i])
    return keras.utils.to_categorical(y, num_classes)


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


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
    # Load the CIFAR10 data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Input image dimensions.
    input_shape = x_train.shape[1:]

    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print('y_train shape:', y_train.shape)

    # Convert class vectors to binary class matrices.
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)
    y_train_sequence = label_preprocess(y_train, down_sampling_ratio)
    y_test_sequence = label_preprocess(y_test, down_sampling_ratio)

    # model = BiLSTM_Sequential_Classification(input_shape=input_shape, classes=num_classes)
    # model = BiLSTM_Deep_Classification(input_shape=input_shape, classes=num_classes)
    # model = BiLSTM_Deep_V_0_1(input_shape=input_shape, classes=num_classes)
    # model = BiLSTM_Deep_V_0_2(input_shape=input_shape, classes=num_classes)
    # model = BiLSTM_Deep_V_0_3(input_shape=input_shape, classes=num_classes)
    model = BiLSTM_Deep_V_0_6(input_shape=input_shape, classes=num_classes)

    #
    # if version == 2:
    #     model = resnet_v2(input_shape=input_shape, depth=depth)
    # else:
    #     model = resnet_v1(input_shape=input_shape, depth=depth)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr_schedule(0)),
                  metrics=['accuracy'])
    model.summary()
    print(model_type)

    # Prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'cifar10-%s-{epoch:03d}-{val_acc:.5f}-{val_loss:.5f}.h5' % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    # Run training, with or without data augmentation.
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train_sequence,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test_sequence),
                  shuffle=True,
                  callbacks=callbacks)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train_sequence, batch_size=batch_size),
                            validation_data=(x_test, y_test_sequence),
                            epochs=epochs, verbose=1, workers=4,
                            callbacks=callbacks)

    # Score trained model.
    # scores = model.evaluate(x_test, y_test, verbose=1)
    acc = test(model, x_test, y_test, down_sampling_ratio)

    # print('Test loss:', scores[0])
    print('Test accuracy:', acc)


if __name__ == '__main__':

    args = define_parser()

    batch_size = 512
    if args.bs:
        batch_size = args.bs

    epochs = 300
    if args.ep:
        epochs = args.ep

    # Training parameters
    data_augmentation = True
    num_classes = 10
    down_sampling_ratio = 1 / 4  # Todo: Important
    input_size = 32
    # Subtracting pixel mean improves accuracy
    subtract_pixel_mean = True

    # Model name, depth and version
    model_type = 'BiLSTM_Deep_V_0_6'  # Todo: Important

    main()
