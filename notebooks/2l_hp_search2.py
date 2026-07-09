import random
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers, metrics
from sklearn.model_selection import train_test_split
import keras_tuner
from keras_tuner import RandomSearch, Hyperband, BayesianOptimization

desc_dim = 887
emb_dim = 1024
dtype = tf.float32
data_path = 'G:/Informatica/IA/enzimas/datasets/Second_Level/'
output_path = 'G:/Informatica/IA/enzimas/models/Second_Level/'
embs_path = data_path + 'valid_emb.npy'
feats_path = data_path + 'valid_feat.npy'
labels_path = data_path + 'y_valid.npy'


def set_reproducibility(seed=42):
    # 1. Fijar semilla en Python core
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)

def load_data():
    global embs_path, feats_path, labels_path
    embs = np.load(embs_path)
    feats = np.load(feats_path)
    y = np.load(labels_path)
    X = np.concatenate([embs, feats], axis=1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_emb = X_train[:,:1024].copy()
    X_train_feat = X_train[:,1024:].copy()
    X_val_emb = X_val[:,:1024].copy()
    X_val_feat = X_val[:,1024:].copy()
    return ((X_train_emb, X_train_feat), y_train), ((X_val_emb, X_val_feat), y_val)


def build_model(hp) -> keras.Model:
    input_desc = layers.Input(shape=(desc_dim,), dtype=dtype)
    input_emb = layers.Input(shape=(emb_dim,), dtype=dtype)
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    activation = hp.Choice('activation', values=['relu', 'swish', 'leaky_relu', 'elu'])
    l2_reg = hp.Float('l2_reg', min_value=1e-5, max_value=1e-2, sampling='log')

    #y = input_emb
    y = input_emb
    units_e = hp.Int(f'emb_units_start', min_value=256, max_value=768, step=64)
    for i in range(hp.Int('emb_layers', 1, 3)):
        y = layers.Dense(units_e // (1 << i), kernel_regularizer=keras.regularizers.l2(l2_reg), dtype=dtype)(y)
        y = layers.BatchNormalization()(y)
        y = layers.Activation(activation)(y)
        y = layers.Dropout(dropout_rate)(y)

    #x = input_desc
    x = input_desc
    units_d = hp.Int(f'desc_units_start', min_value=128, max_value=512, step=64)
    for i in range(hp.Int('desc_layers', 1, 3)):
        x = layers.Dense(units_d // (1 << i), kernel_regularizer=keras.regularizers.l2(l2_reg), dtype=dtype)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        x = layers.Dropout(dropout_rate)(x)

    combined = layers.Concatenate()([y, x])
    
    z = combined
    units_h = hp.Int(f'head_units_start', min_value=64, max_value=256, step=64)
    for i in range(hp.Int('head_layers', 1, 3)):
        z = layers.Dense(units_h // (1 << i), kernel_regularizer=keras.regularizers.l2(l2_reg), dtype=dtype)(z)
        z = layers.BatchNormalization()(z)
        z = layers.Activation(activation)(z)
        z = layers.Dropout(dropout_rate)(z)

    z = layers.Dense(7)(z)
    output = layers.Activation('softmax')(z)
    model = models.Model([input_emb, input_desc], output, name='classifier_2l')
    learning_rate = hp.Float('learning_rate', min_value=1e-6, max_value=1e-3, sampling='log')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=[metrics.AUC(curve='ROC', multi_label=True), metrics.F1Score(average='macro')]
    )
    return model

if __name__ == '__main__':
    set_reproducibility()
    train, valid = load_data()
    tuner = Hyperband(
        build_model,
        objective = "val_loss",
        max_epochs=50,
        factor=3,
        seed=42,
        directory='tunning_SL_no_noise',
        project_name='enzyme_dual_branch'
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=8,
            restore_best_weights=True 
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5,
            patience=3, 
            min_lr=1e-6,
            verbose=1
        )
    ]
    tuner.search(train[0], train[1], validation_data=(valid[0], valid[1]), epochs=50, callbacks = callbacks)
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hp)
    best_model.save(output_path + 'untrained_model_no_noise.keras')


