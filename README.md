# 3D CNN DeepFake Detection using TensorFlow

This repository provides a complete baseline implementation of a **3D Convolutional Neural Network (3D CNN)** for **DeepFake Detection (DFD)** using video-level classification.

The model processes a fixed number of frames from each video and predicts whether the video is **Real** or **Fake**.

---

## 1. Project Overview

DeepFake detection is a binary video classification problem. In this project, each input video is represented as a short temporal sequence of frames. A 3D CNN is used to learn both:

- spatial features from individual frames
- temporal features across consecutive frames

The final output is a probability score:

```text
0 = Real
1 = Fake
```

---

## 2. Repository Structure

```text
3D_CNN_DeepFake_Detection_TensorFlow/
│
├── train_3dcnn_dfd.py          # Main training and evaluation script
├── predict_video.py            # Single-video prediction script
├── requirements.txt            # Python dependencies
├── .gitignore                  # Files ignored by Git
├── LICENSE                     # MIT License
└── README.md                   # Project documentation
```

---

## 3. Dataset Structure

Before training, arrange your dataset as follows:

```text
DFD_Dataset/
│
├── train/
│   ├── real/
│   │   ├── video_001.mp4
│   │   └── video_002.mp4
│   └── fake/
│       ├── video_003.mp4
│       └── video_004.mp4
│
├── val/
│   ├── real/
│   └── fake/
│
└── test/
    ├── real/
    └── fake/
```

Label convention:

```text
real = 0
fake = 1
```

Supported video formats:

```text
.mp4, .avi, .mov, .mkv
```

---

## 4. Installation

Create a new Python environment:

```bash
python -m venv venv
```

Activate it:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 5. Training

Edit the dataset path inside `train_3dcnn_dfd.py`:

```python
DATASET_DIR = "DFD_Dataset"
```

Then run:

```bash
python train_3dcnn_dfd.py
```

The script will:

1. Load videos from the dataset folders.
2. Uniformly sample frames from each video.
3. Train a 3D CNN model.
4. Save the best model using validation AUC.
5. Evaluate the model on the test set.
6. Print accuracy, confusion matrix, and classification report.

---

## 6. Prediction on a Single Video

After training, run:

```bash
python predict_video.py --video path/to/video.mp4 --model best_3dcnn_dfd_model.keras
```

Example output:

```text
Prediction probability: 0.8732
Predicted class: Fake
```

---

## 7. Model Architecture

The implemented baseline model contains:

- 3D convolution layers
- batch normalization
- 3D max pooling
- global average pooling
- dense classification layers
- dropout regularization
- sigmoid output layer

The model is trained using:

```text
Loss: Binary Crossentropy
Optimizer: Adam
Metrics: Accuracy, AUC, Precision, Recall
```

---

## 8. Recommended Settings

For limited GPU memory:

```python
IMG_SIZE = 112
NUM_FRAMES = 16
BATCH_SIZE = 2
```

For stronger training:

```python
IMG_SIZE = 160
NUM_FRAMES = 32
BATCH_SIZE = 4
```

For large GPUs:

```python
IMG_SIZE = 224
NUM_FRAMES = 32
BATCH_SIZE = 4 or 8
```

---

## 9. Notes for Research Use

This repository is intended as a clean baseline for DeepFake Detection. For a stronger academic paper, possible extensions include:

- face detection and face cropping before training
- CNN-LSTM architecture
- 3D ResNet
- EfficientNet + temporal attention
- Vision Transformer / TimeSformer
- Grad-CAM or explainable AI visualization
- cross-dataset validation
- compression robustness evaluation

---

## 10. Citation

If you use this code in your research, you may cite it as:

```bibtex
@misc{3dcnn_dfd_tensorflow,
  title={3D CNN DeepFake Detection using TensorFlow},
  author={Your Name},
  year={2026},
  note={GitHub repository}
}
```

---

## 11. License

This project is released under the MIT License.
