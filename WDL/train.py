#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Project: ctrip_tkt_deep_model
@File   : train.py
@Author : sichenghe(sichenghe@trip.com)
@Time   : 2022/7/28 
"""

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

from model import WideDeep
from data_process.criteo import create_criteo_dataset



if __name__ == '__main__':
    # =============================== GPU ==============================
    # gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
    # print(gpu)
    # If you have GPU, and the value is GPU serial number.
    # os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    # ========================= Hyper Parameters =======================
    # you can modify your file path
    # file = '../dataset/Criteo/train.txt'
    file = '../dataset/criteo/criteo_sampled_data.csv'
    read_part = True
    sample_num = 20000
    test_size = 0.2

    embed_dim = 8
    dnn_dropout = 0.2
    hidden_units = [256, 128, 64]

    learning_rate = 0.001
    batch_size = 128
    epochs = 5

    # ========================== Create dataset =======================
    feature_columns, train, test = create_criteo_dataset(file=file,
                                                         embed_dim=embed_dim,
                                                         read_part=read_part,
                                                         sample_num=sample_num,
                                                         test_size=test_size)
    train_X, train_y = train
    test_X, test_y = test
    # ============================Build Model==========================
    # with mirrored_strategy.scope():
    #     model = WideDeep(feature_columns, hidden_units=hidden_units, dnn_dropout=dnn_dropout)
    #     model.summary()
    #     # ============================Compile============================
    #     model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
    #                   metrics=[AUC()])
    # ============================model checkpoint======================
    # check_path = '../save/wide_deep_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # ==============================Fit==============================

    model = WideDeep(feature_columns, hidden_units=hidden_units, dnn_dropout=dnn_dropout)
    # model.summary()
    # ============================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                      metrics=[AUC()])
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],  # checkpoint
        batch_size=batch_size,
        validation_split=0.1
    )
    # ===========================Test==============================
    loss = model.evaluate(test_X, test_y, batch_size=batch_size)
    print('test loss: %f' % loss[0])
    print('test AUC: %f' % loss[1])
