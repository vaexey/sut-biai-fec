from model import build_model
from model import lr_scheduler
from model import early_stop
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.losses import CategoricalCrossentropy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import utils as ut
import log
import eval

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

matplotlib.use('TkAgg')  # tkinter backend

log_dir = log.create_log_dir()

BATCH_SIZE = 32
# 1e-3
LEARNING_RATE = 1e-3

train_generator = ut.load_data(ut.DATA_TRAIN_PATH, batch_size=BATCH_SIZE)

num_classes = len(train_generator.class_indices)

model = build_model(num_classes)


model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        # loss='categorical_crossentropy',
        metrics=['accuracy'])

# params
EPOCHS = 75
STEPS_PER_EPOCH = train_generator.samples // train_generator.batch_size


# weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)

class_weights_dict = dict(enumerate(class_weights))

# train
history = model.fit(
    train_generator,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    callbacks=[log.get_tensorboard_callback(log_dir), early_stop, lr_scheduler],
    class_weight=class_weights_dict,
    verbose=1,
)

print("Training completed")

# eval

print("Evaluation start")
eval.run(model, log_dir)

# log

log.save_plot(history, log_dir)
log.save_metrics(history, log_dir)
log.save_model(model, log_dir)
log.dump_training_config(
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    log_dir=log_dir,
    learning_rate=LEARNING_RATE
)
log.dump_model_nn(model, log_dir)



# plot acc and loss

plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['loss'], label='loss')
plt.title('Train Metrics')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.show()



