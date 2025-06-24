from model import build_model
from model import lr_scheduler
from model import early_stop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

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
LEARNING_RATE = 1e-2

train_generator = ut.load_data(ut.DATA_TRAIN_PATH, batch_size=BATCH_SIZE)

num_classes = len(train_generator.class_indices)

model = build_model(num_classes)


model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy'])


# params
EPOCHS = 75
STEPS_PER_EPOCH = train_generator.samples // train_generator.batch_size


val_generator = ut.load_val_data(ut.DATA_VAL_PATH, shuffle=False)
# train
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    callbacks=[log.get_tensorboard_callback(log_dir), early_stop, lr_scheduler],
    verbose=1,
    validation_data=val_generator
)

print("Training completed")

# eval

print("Evaluation start")
acc = eval.run(model, log_dir)

# log

log.save_plot(history, log_dir)
log.save_metrics(history, log_dir)
log.save_model(model, log_dir)
log.dump_training_config(
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    log_dir=log_dir,
    learning_rate=LEARNING_RATE,
    acc=acc
)
log.dump_model_nn(model, log_dir)


# plot acc and loss

plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Train Metrics')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.show()
