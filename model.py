from tensorflow.keras import models
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GlobalAveragePooling2D,Dropout, Rescaling, RandomFlip, RandomRotation, Conv2D, BatchNormalization, Dense, Activation, MaxPooling2D, Flatten


def build_model(num_classes):
        num_features = 32

        model = models.Sequential()

        # module 1
        model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), input_shape=(48, 48, 1), data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # module 2
        model.add(Conv2D(2*num_features, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(2*num_features, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # module 3
        model.add(Conv2D(num_features, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(num_features, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # flatten
        model.add(Flatten())

        # dense 1
        model.add(Dense(2*2*2*num_features))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # dense 2
        model.add(Dense(2*2*num_features))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # dense 3
        model.add(Dense(2*num_features))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # output layer
        model.add(Dense(num_classes, activation='softmax'))

        return model


lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='loss',
    patience=8,
    restore_best_weights=True,
    verbose=1
)
