import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow_addons as tfa
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from tqdm.notebook import tqdm


train_features = pd.read_csv('../input/lish-moa/train_features.csv')
train_targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
test_features = pd.read_csv('../input/lish-moa/test_features.csv')

submission = pd.read_csv('../input/lish-moa/sample_submission.csv')

def preprocess(df):
    df = df.copy()
    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})
    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})
    del df['sig_id']
    return df

train = preprocess(train_features)
test = preprocess(test_features)

del train_targets['sig_id']

train_targets = train_targets.loc[train['cp_type']==0].reset_index(drop=True)
train = train.loc[train['cp_type']==0].reset_index(drop=True)


def create_model(num_columns):
    model = tf.keras.Sequential([
    tf.keras.layers.Input(num_columns),
    tf.keras.layers.BatchNormalization(),
    tfa.layers.WeightNormalization(tf.keras.layers.Dense(2048, activation="relu")),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tfa.layers.WeightNormalization(tf.keras.layers.Dense(2048, activation="relu")),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tfa.layers.WeightNormalization(tf.keras.layers.Dense(206, activation="sigmoid"))
    ])
    model.compile(optimizer=tfa.optimizers.Lookahead(tf.optimizers.Adam(), sync_period=10),
                  loss='binary_crossentropy'
                  )
    return model

def metric(y_true, y_pred):
    metrics = []
    for _target in train_targets.columns:
        metrics.append(log_loss(y_true.loc[:, _target], y_pred.loc[:, _target], labels=[0,1]))
    return np.mean(metrics)

N_STARTS =3
tf.random.set_seed(42)

res = train_targets.copy()
submission.loc[:, train_targets.columns] = 0
res.loc[:, train_targets.columns] = 0

for seed in range(N_STARTS):
    for n, (tr, te) in enumerate(KFold(n_splits=7, random_state=seed, shuffle=True).split(train_targets)):
        print(f'Fold Started - {n}')

        model = create_model(len(train.columns))
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, epsilon=1e-3, mode='max')
        model.fit(train.values[tr],
                  train_targets.values[tr],
                  validation_data=(train.values[te], train_targets.values[te]),
                  epochs=25, batch_size=64,shuffle=True,
                  callbacks=[reduce_lr_loss], verbose=2
                 )

        test_predict = model.predict(test.values)
        val_predict = model.predict(train.values[te])
        submission.loc[:, train_targets.columns] += test_predict
        res.loc[te, train_targets.columns] += val_predict
        print('Fold Completed - ',n)

submission.loc[:, train_targets.columns] /= ((n+1) * N_STARTS)
res.loc[:, train_targets.columns] /= N_STARTS

submission.loc[test['cp_type']==1, train_targets.columns] = 0

submission.to_csv('submission.csv', index=False)

print(submission.head())