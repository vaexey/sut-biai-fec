from model import build_model
import matplotlib.pyplot as plt
import matplotlib
import utils as ut
import log
matplotlib.use('TkAgg')  # tkiner backend

log_dir = log.create_log_dir()

train_generator = ut.load_data(ut.DATA_TRAIN_PATH)

num_classes = len(train_generator.class_indices)

model = build_model(num_classes)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=30,
    callbacks=[log.get_tensorboard_callback(log_dir)],
)

# eval
print("training complete")

log.save_plot(history, log_dir)
log.save_metrics(history, log_dir)
log.save_model(model, log_dir)


# plot acc and loss
plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['loss'], label='loss')
plt.title('Train Metrics')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.show()



