"""
3D CNN DeepFake Detection using TensorFlow

Author: Your Name
Task: Binary video classification
Classes:
    0 = Real
    1 = Fake

Dataset structure:

DFD_Dataset/
    train/
        real/
        fake/
    val/
        real/
        fake/
    test/
        real/
        fake/
"""

import os
import cv2
import glob
import argparse
import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

IMG_SIZE = 112
NUM_FRAMES = 32
CHANNELS = 3

BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 1e-4

VIDEO_EXTENSIONS = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# GPU CONFIGURATION
# ============================================================

def enable_gpu_memory_growth():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("[INFO] GPU memory growth enabled.")
        except RuntimeError as error:
            print("[WARNING]", error)


# ============================================================
# VIDEO LOADING
# ============================================================

def load_video(video_path, num_frames=NUM_FRAMES, img_size=IMG_SIZE):
    """
    Load a video and uniformly sample a fixed number of frames.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to sample.
        img_size: Height and width of resized frames.

    Returns:
        NumPy array with shape:
        (num_frames, img_size, img_size, 3)
    """

    video_path = video_path.decode("utf-8")
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if total_frames <= 0:
        cap.release()
        return np.zeros((num_frames, img_size, img_size, 3), dtype=np.float32)

    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()

        if not success:
            frame = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size))

        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)

    cap.release()

    frames = np.array(frames, dtype=np.float32)

    if frames.shape[0] < num_frames:
        pad_length = num_frames - frames.shape[0]
        padding = np.zeros((pad_length, img_size, img_size, 3), dtype=np.float32)
        frames = np.concatenate([frames, padding], axis=0)

    return frames


def tf_load_video(video_path, label):
    """
    TensorFlow wrapper for OpenCV video loading.
    """

    video = tf.numpy_function(
        func=load_video,
        inp=[video_path],
        Tout=tf.float32
    )

    video.set_shape((NUM_FRAMES, IMG_SIZE, IMG_SIZE, CHANNELS))
    label = tf.cast(label, tf.float32)

    return video, label


# ============================================================
# DATASET CREATION
# ============================================================

def collect_video_paths(split_dir):
    """
    Collect video paths and labels from a split folder.

    Expected folder:
        split_dir/
            real/
            fake/
    """

    video_paths = []
    labels = []

    real_dir = os.path.join(split_dir, "real")
    fake_dir = os.path.join(split_dir, "fake")

    for extension in VIDEO_EXTENSIONS:
        real_files = glob.glob(os.path.join(real_dir, extension))
        fake_files = glob.glob(os.path.join(fake_dir, extension))

        video_paths.extend(real_files)
        labels.extend([0] * len(real_files))

        video_paths.extend(fake_files)
        labels.extend([1] * len(fake_files))

    video_paths = np.array(video_paths)
    labels = np.array(labels)

    return video_paths, labels


def create_dataset(video_paths, labels, training=True):
    """
    Create TensorFlow dataset from video paths and labels.
    """

    dataset = tf.data.Dataset.from_tensor_slices((video_paths, labels))

    if training:
        dataset = dataset.shuffle(buffer_size=len(video_paths), reshuffle_each_iteration=True)

    dataset = dataset.map(tf_load_video, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


# ============================================================
# MODEL
# ============================================================

def build_3d_cnn(input_shape=(NUM_FRAMES, IMG_SIZE, IMG_SIZE, CHANNELS)):
    """
    Build a baseline 3D CNN for binary video classification.
    """

    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv3D(32, kernel_size=(3, 3, 3), padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(x)

    x = tf.keras.layers.Conv3D(64, kernel_size=(3, 3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

    x = tf.keras.layers.Conv3D(128, kernel_size=(3, 3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

    x = tf.keras.layers.Conv3D(256, kernel_size=(3, 3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

    x = tf.keras.layers.GlobalAveragePooling3D()(x)

    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


# ============================================================
# TRAINING AND EVALUATION
# ============================================================

def train_and_evaluate(dataset_dir):
    enable_gpu_memory_growth()

    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")
    test_dir = os.path.join(dataset_dir, "test")

    train_paths, train_labels = collect_video_paths(train_dir)
    val_paths, val_labels = collect_video_paths(val_dir)
    test_paths, test_labels = collect_video_paths(test_dir)

    print("[INFO] Training videos:", len(train_paths))
    print("[INFO] Validation videos:", len(val_paths))
    print("[INFO] Testing videos:", len(test_paths))

    if len(train_paths) == 0:
        raise ValueError("No training videos found. Please check your dataset path and folder structure.")

    train_dataset = create_dataset(train_paths, train_labels, training=True)
    val_dataset = create_dataset(val_paths, val_labels, training=False)
    test_dataset = create_dataset(test_paths, test_labels, training=False)

    model = build_3d_cnn()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath="best_3dcnn_dfd_model.keras",
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    model.save("final_3dcnn_dfd_model.keras")
    print("[INFO] Final model saved as final_3dcnn_dfd_model.keras")

    print("\n[INFO] Evaluating on test set...")
    test_results = model.evaluate(test_dataset)
    print("[INFO] Test results:", test_results)

    y_true = []
    y_pred = []

    for videos, labels in test_dataset:
        probabilities = model.predict(videos)
        predictions = (probabilities >= 0.5).astype(int).flatten()

        y_true.extend(labels.numpy().astype(int))
        y_pred.extend(predictions)

    print("\nAccuracy:", accuracy_score(y_true, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))

    return history


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D CNN for DeepFake Detection")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="DFD_Dataset",
        help="Path to dataset folder containing train, val, and test folders"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(args.dataset_dir)
