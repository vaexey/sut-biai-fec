from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

DATA_DIR = "data_no_disgust"
DATA_TRAIN_PATH = f'{DATA_DIR}/train'
DATA_VAL_PATH = f'{DATA_DIR}/test'
IMG_SIZE = 48


def get_meta():
    class_names = os.listdir(f'{DATA_DIR}/train')
    return class_names


def load_data(train_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=32, shuffle=True):
    # train_datagen = ImageDataGenerator(
    #     rescale=1./255,  # normalize
    #     rotation_range=20,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     horizontal_flip=True,
    #     fill_mode='nearest'
    # )

    train_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=shuffle
    )

    return train_generator
