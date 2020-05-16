import csv
import tensorflow as tf

class LogCalllback(tf.keras.callbacks.Callback):
    def __init__(self, prefix, log_path):
        self.prefix = prefix
        self.log_path = log_path

    def on_epoch_end(self, epoch, logs=None):
        with open(self.log_path, 'a+') as log:
            fieldnames = ['model_prefix', 'epoch', 'train_loss', 'train_acc' ,'val_loss', 'val_acc']
            writer = csv.DictWriter(log, fieldnames=fieldnames)
            writer.writerow({
                    'model_prefix': self.prefix,
                    'epoch': epoch,
                    'train_loss':logs['loss'],
                    'train_acc':logs['categorical_accuracy'],
                    'val_loss':logs['val_loss'],
                    'val_acc':logs['val_categorical_accuracy']
                 })