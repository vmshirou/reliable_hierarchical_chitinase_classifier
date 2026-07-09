import numpy as np
import os
import random
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers, losses, metrics

"""
Global variables
"""



current_dir = Path(__file__).parent.parent.parent
train_emb_dir = current_dir / 'datasets/Second_Level/train_emb.npy'
train_feat_dir = current_dir / 'datasets/Second_Level/train_feat.npy'
valid_emb_dir = current_dir / 'datasets/Second_Level/valid_emb.npy'
valid_feat_dir = current_dir / 'datasets/Second_Level/valid_feat.npy'
y_train_dir = current_dir / 'datasets/Second_Level/y_train.npy'
y_valid_dir = current_dir / 'datasets/Second_Level/y_valid.npy'
history_dir = Path(__file__).parent / 'history.csv'
model_dir = current_dir / 'models/Second_Level/untrained_model.keras'
model_output = Path(__file__).parent / 'model.keras'
desc_dim = 887
emb_dim = 1024
batch_size = 32
learning_rate = 0.0001
epoch = 100
layers_first_in = (512, 256, 128)
layers_second_in = (512, 256, 128)
layers_output = (128, 32, 7)
middle_activation = layers.LeakyReLU
last_activation = layers.Activation
last_activation_name = 'softmax'
loss = losses.categorical_crossentropy
tf.random.set_seed(42)
tf.keras.mixed_precision.set_global_policy('float32')
optimizer = optimizers.Adam(learning_rate, clipnorm=1.0)
def set_reproducibility(seed=42):
    # 1. Fijar semilla en Python core
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)
set_reproducibility()

def load_data():
    train_emb = np.load(train_emb_dir)
    train_feat = np.load(train_feat_dir)
    valid_emb = np.load(valid_emb_dir)
    valid_feat = np.load(valid_feat_dir)
    y_train = np.load(y_train_dir)
    y_valid = np.load(y_valid_dir)
    train = tf.data.Dataset.from_tensor_slices(
        ((train_emb.astype('float32'), train_feat.astype('float32')), y_train)
    ).batch(batch_size)
    valid = tf.data.Dataset.from_tensor_slices(
        ((valid_emb.astype('float32'), valid_feat.astype('float32')), y_valid)
    ).batch(batch_size)
    return train, valid


def build_model(units_1: tuple, units_2: tuple, units_end: tuple, activation, activation_end, activation_end_name: str, dropout_rate: float = 0.5, dtype: tf.DType = tf.float32) -> keras.Model:
    input_desc = layers.Input(shape=(desc_dim,), dtype=dtype)
    input_emb = layers.Input(shape=(emb_dim,), dtype=dtype)
    x = input_desc
    y = input_emb
    for unit in units_1:
        x = layers.Dense(unit, dtype=dtype)(x)
        x = activation()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
    for unit in units_2:
        x = layers.Dense(unit, dtype=dtype)(x)
        x = activation()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
    x = layers.Concatenate()([x, y])
    for i in range(len(units_end) - 1):
        x = layers.Dense(units_end[i], dtype=dtype)(x)
        x = activation()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(units_end[-1])(x)
    output = activation_end(activation_end_name)(x)
    model = models.Model([input_emb, input_desc], output, name='classifier_1l')
    return model

def reset_weigths(model: models.Model):
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.assign(layer.kernel_initializer(shape=layer.kernel.shape, dtype=layer.kernel.dtype))
        if hasattr(layer, 'bias_initializer') and layer.use_bias:
            layer.bias.assign(layer.bias_initializer(shape=layer.bias.shape, dtype=layer.bias.dtype))
        if hasattr(layer, 'moving_mean'):
            layer.moving_mean.assign(tf.zeros_like(layer.moving_mean))
        if hasattr(layer, 'moving_variance'):
            layer.moving_variance.assign(tf.ones_like(layer.moving_variance))

def load_untrained_model():
    global model_dir
    model = models.load_model(str(model_dir))
    reset_weigths(model)
    return model



from typing import Any


class Classifier(keras.Model):
    def __init__(self, model: keras.Model, loss, **kwargs: Any) -> None:
        super(Classifier, self).__init__(**kwargs)
        self.model = model
        self.current_loss = loss
        self.total_loss_tracker = metrics.Mean(name="loss")
        self.auc_tracker = metrics.AUC(curve='ROC', multi_label=True)
        self.f1_tracker = metrics.F1Score(average='macro')
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.auc_tracker,
            self.f1_tracker
        ]
    
    def call(self, inputs) -> Any:
        return self.model(inputs)
    
    def train_step(self, data):
        X, y = data
        with tf.GradientTape() as tape:
            pred = self(X)
            loss = self.current_loss(y, pred)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.total_loss_tracker.update_state(loss)
        pre_ind = tf.argmax(pred, axis=-1)
        one_hot_preds = tf.one_hot(pre_ind, pred.shape[-1])
        self.auc_tracker.update_state(y, one_hot_preds)
        self.f1_tracker.update_state(y, one_hot_preds)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        X, y = data
        pred = self(X)
        loss = self.current_loss(y, pred)
        self.total_loss_tracker.update_state(loss)
        pre_ind = tf.argmax(pred, axis=-1)
        one_hot_preds = tf.one_hot(pre_ind, pred.shape[-1])
        self.auc_tracker.update_state(y, one_hot_preds)
        self.f1_tracker.update_state(y, one_hot_preds)
        return {m.name: m.result() for m in self.metrics}

def train(train_data: tf.data.Dataset, valid_data: tf.data.Dataset, load_model=False) -> None:
    ea = callbacks.EarlyStopping(
        monitor='val_f1_score',
	mode='max',
        patience=8,
        restore_best_weights=True
    )
    logger = callbacks.CSVLogger(str(history_dir))
    rlr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_f1_score',
	mode='max', 
        factor=0.5,
        patience=3, 
        min_lr=1e-6,
        verbose=1
    )
    model = None
    if load_model:
        model = load_untrained_model()
    else:
        model = build_model(layers_first_in, layers_second_in, layers_output, middle_activation, last_activation, last_activation_name)
    classifier = Classifier(model, loss)
    classifier.compile(optimizer)
    classifier.fit(train_data, batch_size=batch_size, epochs=epoch, callbacks=[ea, logger, rlr], validation_data=valid_data)
    classifier.model.save(str(model_output))
    print('Training process completed')


if __name__ == '__main__':
    try:
        train_data, valid_data = load_data()
        train(train_data, valid_data, load_model=True)
    except Exception as e:
        print(f'Something went wrong: {e}')