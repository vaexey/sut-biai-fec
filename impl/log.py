import os
import json
import io
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.callbacks import TensorBoard


def create_log_dir(base_dir='logs'):
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(base_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def get_tensorboard_callback(log_dir):
    return TensorBoard(log_dir=log_dir, histogram_freq=1)


def dump_training_config(
    batch_size,
    epochs,
    steps_per_epoch,
    log_dir,
    learning_rate,
    acc
):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f'training_config_{timestamp}.json')

    config = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'steps_per_epoch': steps_per_epoch,
        'acc': acc
    }

    with open(log_path, 'w') as f:
        json.dump(config, f, indent=4)


def save_plot(history, log_dir, filename='training_plot.png'):
    plt.figure()
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'], label='train_accuracy')
        plt.plot(history.history.get('val_accuracy', []), label='val_accuracy')
    if 'loss' in history.history:
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history.get('val_loss', []), label='val_loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(log_dir, filename))
    plt.close()


def save_metrics(history, log_dir):
    with open(os.path.join(log_dir, 'training_metrics.json'), 'w') as f:
        json.dump(history.history, f, indent=4)

    pd.DataFrame(history.history).to_csv(
        os.path.join(log_dir, 'training_metrics.csv'),
        index=False
    )


def save_model(model, log_dir, filename='model.keras'):
    model.save(os.path.join(log_dir, filename))


def dump_model_nn(model, log_dir):
    stringio = io.StringIO()
    sys.stdout = stringio
    model.summary()
    sys.stdout = sys.__stdout__
    log_path = os.path.join(log_dir, "model_nn.txt")
    with open(log_path, 'w') as f:
        f.write(stringio.getvalue())

    print(f"Model summary saved to {log_path}")
