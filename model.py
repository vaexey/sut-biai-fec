from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Rescaling, RandomFlip, RandomRotation, Conv2D, Input, BatchNormalization, Dense, Activation, MaxPooling2D, Flatten


# def build_model(num_classes):
#     model = models.Sequential([
#         layers.Input(shape=(48, 48, 1)),
#         layers.Conv2D(32, (3, 3), activation='relu'),
#         layers.MaxPooling2D(),
#         layers.Conv2D(64, (3, 3), activation='relu'),
#         layers.MaxPooling2D(),
#         layers.Conv2D(128, (3, 3), activation='relu'),
#         layers.Flatten(),
#         layers.Dense(128, activation='relu'),
#         layers.Dropout(0.5),
#         layers.Dense(num_classes, activation='softmax'),
#     ])
#     return model


# def build_model(num_classes):
#     model = models.Sequential([
#         layers.Input(shape=(48, 48, 1)),
#
#         layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D(),
#
#         layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D(),
#
#         layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D(),
#
#         layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D(),
#
#         layers.GlobalAveragePooling2D(),
#         layers.Dense(128, activation='relu'),
#         layers.Dropout(0.5),
#         layers.Dense(num_classes, activation='softmax'),
#     ])
#     return model

# sigmoid no batch normalization
# def build_model(num_classes):
#     model = models.Sequential([
#         layers.Input(shape=(48, 48, 1)),
#
#         layers.Conv2D(32, (3, 3), padding='same', strides=(1, 1),
#                       kernel_regularizer=regularizers.l2(0.001)),
#         layers.Dropout(0.1),
#         layers.Activation('relu'),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#
#         layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1),
#                       kernel_regularizer=regularizers.l2(0.001)),
#         layers.Dropout(0.1),
#         layers.Activation('relu'),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#
#         layers.Conv2D(128, (3, 3), padding='same', strides=(1, 1),
#                       kernel_regularizer=regularizers.l2(0.001)),
#         layers.Dropout(0.1),
#         layers.Activation('relu'),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#
#         layers.Conv2D(256, (3, 3), padding='same', strides=(1, 1),
#                       kernel_regularizer=regularizers.l2(0.001)),
#         layers.Dropout(0.1),
#         layers.Activation('relu'),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#
#         layers.Flatten(),
#         layers.Dense(128, activation='relu'),
#         layers.Dropout(0.2),
#         layers.Dense(num_classes, activation='sigmoid')
#     ])
#     return model


#softmanx batch normalization
# def build_model(num_classes):
#     model = models.Sequential([
#         layers.Input(shape=(48, 48, 1)),
#
#         layers.Conv2D(32, (3, 3), padding='same', strides=(1, 1),
#                       kernel_regularizer=regularizers.l2(0.001)),
#         layers.BatchNormalization(),
#         layers.Activation('relu'),
#         layers.Dropout(0.1),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#
#         layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1),
#                       kernel_regularizer=regularizers.l2(0.001)),
#         layers.BatchNormalization(),
#         layers.Activation('relu'),
#         layers.Dropout(0.1),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#
#         layers.Conv2D(128, (3, 3), padding='same', strides=(1, 1),
#                       kernel_regularizer=regularizers.l2(0.001)),
#         layers.BatchNormalization(),
#         layers.Activation('relu'),
#         layers.Dropout(0.1),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#
#         layers.Conv2D(256, (3, 3), padding='same', strides=(1, 1),
#                       kernel_regularizer=regularizers.l2(0.001)),
#         layers.BatchNormalization(),
#         layers.Activation('relu'),
#         layers.Dropout(0.1),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#
#         layers.Flatten(),
#         layers.Dense(128),
#         layers.BatchNormalization(),
#         layers.Activation('relu'),
#         layers.Dropout(0.2),
#         layers.Dense(num_classes, activation='softmax')
#     ])
#     return model

#some paper
# def build_model(num_classes):
#     model = models.Sequential([
#         layers.Input(shape=(48, 48, 1)),
#
#         # Conv Layer 1
#         layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
#         # Conv Layer 2
#         layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Dropout(0.25),
#
#         # Conv Layer 3
#         layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Dropout(0.25),
#
#         # Conv Layer 4
#         layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#
#         # Fully connected
#         layers.Flatten(),
#         layers.Dense(1024, activation='relu'),
#         layers.Dense(num_classes, activation='softmax')  # Output layer for 7-class classification
#     ])
#     return model

# with augmentatin in model
# def build_model(num_classes=7, use_augmentation=True):
#     model = models.Sequential()
#
#     # Optional Data Augmentation
#     if use_augmentation:
#         model.add(RandomFlip("horizontal"))
#         model.add(RandomRotation(0.1))
#
#     # Normalize input pixels to [0, 1]
#     model.add(Rescaling(1./255, input_shape=(48, 48, 1)))
#
#     # Conv Block 1
#     model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
#     model.add(layers.BatchNormalization())
#
#     # Conv Block 2
#     model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
#     model.add(layers.BatchNormalization())
#     model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#     model.add(layers.Dropout(0.25))
#
#     # Conv Block 3
#     model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
#     model.add(layers.BatchNormalization())
#     model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#     model.add(layers.Dropout(0.25))
#
#     # Conv Block 4
#     model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
#     model.add(layers.BatchNormalization())
#     model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#
#     # Fully Connected
#     model.add(layers.Flatten())
#     model.add(layers.Dense(1024, activation='relu'))
#     model.add(layers.BatchNormalization())
#     model.add(layers.Dropout(0.5))  # Stronger dropout for FC layer
#
#     # Output Layer
#     model.add(layers.Dense(num_classes, activation='softmax'))
#
#     return model

#kaggle
def build_model(num_classes):
        num_features = 32

        model = models.Sequential()

        #module 1
        model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), input_shape=(48, 48, 1), data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        #module 2
        model.add(Conv2D(2*num_features, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(2*num_features, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        #module 3
        model.add(Conv2D(num_features, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(num_features, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        #flatten
        model.add(Flatten())

        #dense 1
        model.add(Dense(2*2*2*num_features))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        #dense 2
        model.add(Dense(2*2*num_features))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        #dense 3
        model.add(Dense(2*num_features))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        #output layer
        model.add(Dense(num_classes, activation='softmax'))

        return model

lr_scheduler = ReduceLROnPlateau(
    monitor='loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

