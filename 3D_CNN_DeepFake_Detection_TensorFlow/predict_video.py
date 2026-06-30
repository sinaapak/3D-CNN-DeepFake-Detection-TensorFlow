"""
Single video prediction script for 3D CNN DeepFake Detection.

Usage:
    python predict_video.py --video path/to/video.mp4 --model best_3dcnn_dfd_model.keras
"""

import cv2
import argparse
import numpy as np
import tensorflow as tf


IMG_SIZE = 112
NUM_FRAMES = 32
CHANNELS = 3


def load_video_for_prediction(video_path, num_frames=NUM_FRAMES, img_size=IMG_SIZE):
    """
    Load and preprocess a single video for prediction.
    """

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []

    if total_frames <= 0:
        cap.release()
        return np.zeros((1, num_frames, img_size, img_size, 3), dtype=np.float32)

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

    frames = np.expand_dims(frames, axis=0)

    return frames


def predict(video_path, model_path):
    model = tf.keras.models.load_model(model_path)

    video_tensor = load_video_for_prediction(video_path)

    probability = model.predict(video_tensor)[0][0]

    predicted_class = "Fake" if probability >= 0.5 else "Real"

    print("Prediction probability:", float(probability))
    print("Predicted class:", predicted_class)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict whether a video is Real or Fake")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--model", type=str, required=True, help="Path to trained Keras model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args.video, args.model)
