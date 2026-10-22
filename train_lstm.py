import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --------------------------------------------------
# 1. Reproducibility and configuration
# --------------------------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_FILE = "Datasets/synthetic_lstm_windows.npz"
MODEL_FILE = "NewModels/lstm_heading.keras"
SCALER_FILE = "NewModels/lstm_heading_scaler.pkl"
RESULTS_FILE = "NewModels/lstm_heading_results.json"

EPOCHS = 40
BATCH_SIZE = 128

# --------------------------------------------------
# 2. Load data
# --------------------------------------------------
data = np.load(DATA_FILE, allow_pickle=False)

X = data["X"].astype(np.float32)
y = data["y"].astype(np.int64)
groups = data["trip_ids"].astype(str)

if X.ndim != 3 or X.shape[2] != 3:
    raise ValueError(
        f"Expected X shape (samples, timesteps, 3), got {X.shape}. "
        "Regenerate the heading dataset first."
    )

if not (len(X) == len(y) == len(groups)):
    raise ValueError("X, y, and trip_ids have different lengths.")

if len(np.unique(y)) != 2:
    raise ValueError("Expected binary labels 0 and 1.")

print("X shape:", X.shape)
print("Label counts:", dict(zip(*np.unique(y, return_counts=True))))
print("Unique trips:", len(np.unique(groups)))

# --------------------------------------------------
# 3. Split by complete trips: 60% train, 20% val, 20% test
# --------------------------------------------------
split1 = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=SEED
)
trainval_idx, test_idx = next(split1.split(X, y, groups))

split2 = GroupShuffleSplit(
    n_splits=1, test_size=0.25, random_state=SEED
)
train_rel, val_rel = next(
    split2.split(
        X[trainval_idx],
        y[trainval_idx],
        groups[trainval_idx]
    )
)

train_idx = trainval_idx[train_rel]
val_idx = trainval_idx[val_rel]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

train_trips = set(groups[train_idx])
val_trips = set(groups[val_idx])
test_trips = set(groups[test_idx])

assert train_trips.isdisjoint(val_trips)
assert train_trips.isdisjoint(test_trips)
assert val_trips.isdisjoint(test_trips)

print("Trips — train:", len(train_trips))
print("Trips — validation:", len(val_trips))
print("Trips — test:", len(test_trips))
print("Windows — train:", len(train_idx))
print("Windows — validation:", len(val_idx))
print("Windows — test:", len(test_idx))

# --------------------------------------------------
# 4. Fit scaler on training data only
# --------------------------------------------------
n_features = X.shape[2]

scaler = StandardScaler()
scaler.fit(X_train.reshape(-1, n_features))

def scale_windows(arr):
    shape = arr.shape
    scaled = scaler.transform(
        arr.reshape(-1, n_features)
    )
    return scaled.reshape(shape).astype(np.float32)

X_train = scale_windows(X_train)
X_val = scale_windows(X_val)
X_test = scale_windows(X_test)

# --------------------------------------------------
# 5. Build LSTM model
# --------------------------------------------------
model = Sequential([
    Input(shape=(X.shape[1], n_features)),
    LSTM(64, return_sequences=True),
    Dropout(0.25),
    LSTM(32),
    Dropout(0.20),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

model.summary()

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    ),
]

# --------------------------------------------------
# 6. Train
# --------------------------------------------------
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    shuffle=True,
    verbose=1,
)

# --------------------------------------------------
# 7. Evaluate on held-out trips
# --------------------------------------------------
probabilities = model.predict(
    X_test, batch_size=BATCH_SIZE, verbose=0
).ravel()

predictions = (probabilities >= 0.2).astype(np.int64)

print("\nLSTM TEST RESULTS")
print("Accuracy :", accuracy_score(y_test, predictions))
print("Precision:", precision_score(
    y_test, predictions, zero_division=0
))
print("Recall   :", recall_score(
    y_test, predictions, zero_division=0
))
print("F1-score :", f1_score(
    y_test, predictions, zero_division=0
))
print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification report:\n", classification_report(
    y_test, predictions, zero_division=0
))

results = {
    "model": "LSTM",
    "features": ["vehicle_speed", "heading_sin", "heading_cos"],
    "input_shape": list(X.shape[1:]),
    "train_trips": len(train_trips),
    "validation_trips": len(val_trips),
    "test_trips": len(test_trips),
    "test_windows": len(y_test),
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision": float(precision_score(
        y_test, predictions, zero_division=0
    )),
    "recall": float(recall_score(
        y_test, predictions, zero_division=0
    )),
    "f1": float(f1_score(
        y_test, predictions, zero_division=0
    )),
    "roc_auc": (
        float(roc_auc_score(y_test, probabilities))
        if len(np.unique(y_test)) == 2 else None
    ),
    "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
}

# Save under NEW filenames; original model files are preserved.
model.save(MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

pd.DataFrame(history.history).to_csv(
    "lstm_heading_history.csv", index=False
)

print("\nSaved model:", MODEL_FILE)
print("Saved scaler:", SCALER_FILE)
print("Saved metrics:", RESULTS_FILE)
