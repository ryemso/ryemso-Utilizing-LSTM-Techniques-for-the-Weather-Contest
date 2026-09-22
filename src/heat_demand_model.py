"""Heat demand forecasting model utilities.

Cleaned from the original experiment notebook so the modeling pipeline can be
reviewed independently from Colab-specific data loading code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Attention,
    BatchNormalization,
    Bidirectional,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Input,
    LSTM,
    LayerNormalization,
    MaxPooling1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


TIMESTEPS = 124
LSTM_UNITS = 64
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 32
PATIENCE = 5

BASE_FEATURES = [
    "기온",
    "풍향",
    "풍속",
    "일강수량",
    "시간 강수량",
    "상대 습도",
    "일사량",
    "체감온도",
    "시각",
    "요일1",
]
TARGET = "열수요"


def create_sequences(X, y=None, timesteps: int = TIMESTEPS):
    """Convert tabular time-series arrays into sliding-window sequences."""
    Xs, ys = [], []
    for i in range(timesteps, len(X)):
        Xs.append(X[i - timesteps : i])
        if y is not None:
            ys.append(y[i])

    X_seq = np.asarray(Xs)
    y_seq = np.asarray(ys) if y is not None else None
    return X_seq, y_seq


def preprocess_data(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """One-hot encode branches and fit scalers on training data only."""
    train_enc = pd.get_dummies(train_df, columns=["지사명"])
    val_enc = pd.get_dummies(val_df, columns=["지사명"])
    val_enc = val_enc.reindex(columns=train_enc.columns, fill_value=0)

    features = [
        col
        for col in train_enc.columns
        if col in BASE_FEATURES or col.startswith("지사명_")
    ]

    X_train = train_enc[features]
    X_val = val_enc[features]
    y_train = train_enc[TARGET].to_numpy()
    y_val = val_enc[TARGET].to_numpy()

    x_scaler = MinMaxScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_val_scaled = x_scaler.transform(X_val)

    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).ravel()

    return (
        X_train_scaled,
        X_val_scaled,
        y_train_scaled,
        y_val_scaled,
        x_scaler,
        y_scaler,
        val_enc,
    )


def build_model(input_shape):
    """CNN + BiLSTM + self-attention regression model."""
    inputs = Input(shape=input_shape)

    x = Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Bidirectional(LSTM(LSTM_UNITS // 2, return_sequences=True))(x)
    x = Dropout(DROPOUT_RATE * 0.5)(x)

    attention = Attention()([x, x])
    x = Concatenate()([x, attention])

    x = Dense(64, activation="relu")(x)
    x = LayerNormalization()(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(1)(x)

    outputs = x[:, -1, :]

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Prepare sequences, train the model, and return preprocessing artifacts."""
    (
        X_train_scaled,
        X_val_scaled,
        y_train_scaled,
        y_val_scaled,
        x_scaler,
        y_scaler,
        val_processed,
    ) = preprocess_data(train_df, val_df)

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled)

    model = build_model((TIMESTEPS, X_train_seq.shape[2]))

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        X_train_seq,
        y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    return (
        model,
        history,
        x_scaler,
        y_scaler,
        val_processed,
        (X_val_seq, y_val_seq),
    )


def evaluate_model(model, X_val_seq, y_val_seq, y_scaler):
    """Evaluate on the original target scale."""
    y_pred_scaled = model.predict(X_val_seq).ravel()
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = y_scaler.inverse_transform(y_val_seq.reshape(-1, 1)).ravel()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "predictions": y_pred,
    }
