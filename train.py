from model import build_model
from model import lr_scheduler
from model import early_stop
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib
import utils as ut
import log

matplotlib.use('TkAgg')  # tkinter backend

log_dir = log.create_log_dir()

train_generator = ut.load_data(ut.DATA_TRAIN_PATH)

num_classes = len(train_generator.class_indices)

model = build_model(num_classes)

model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

# params
EPOCHS = 40
STEPS_PER_EPOCH = train_generator.samples // train_generator.batch_size

# train
history = model.fit(
    train_generator,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    callbacks=[log.get_tensorboard_callback(log_dir), early_stop, lr_scheduler],
    verbose=1,
)

# eval
print("training complete")

log.save_plot(history, log_dir)
log.save_metrics(history, log_dir)
log.save_model(model, log_dir)
log.dump_training_config(
    batch_size=ut.BATCH_SIZE,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    log_dir=log_dir
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



