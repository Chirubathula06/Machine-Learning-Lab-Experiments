import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

tf.random.set_seed(42)
np.random.seed(42)

def run_ann(name, X, y, classes, loss):
    print(f"\n========== {name} ==========")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(X.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(classes, activation='softmax' if classes>1 else 'sigmoid')
    ])

    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=30, verbose=0)

    predictions = model.predict(X_test, verbose=0)

    if classes > 1:
        y_pred = np.argmax(predictions, axis=1)
    else:
        y_pred = (predictions > 0.5).astype(int).flatten()

    print("\nActual Output   Predicted Output")
    for i in range(10):
        print(f"{y_test[i]} \t\t {y_pred[i]}")

    acc = np.mean(y_test == y_pred)
    print("\nAccuracy:", round(acc,4))


# Run applications
run_ann("Healthcare – Cancer Prediction",
        load_breast_cancer().data,
        load_breast_cancer().target,
        1, 'binary_crossentropy')

run_ann("Education – Iris Classification",
        load_iris().data,
        load_iris().target,
        3, 'sparse_categorical_crossentropy')

run_ann("Agriculture – Wine Classification",
        load_wine().data,
        load_wine().target,
        3, 'sparse_categorical_crossentropy')

run_ann("Handwriting – Digit Recognition",
        load_digits().data,
        load_digits().target,
        10, 'sparse_categorical_crossentropy')
