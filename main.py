import utils as ut
import tensorflow as tf

if __name__ == "__main__":
    ut.get_meta()
    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
